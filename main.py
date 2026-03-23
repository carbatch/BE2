"""
BE2/main.py

FastAPI 서버 — Stable Diffusion 1.5 이미지 생성
모델: runwayml/stable-diffusion-v1-5

실행 방법:
  pip install -r requirements.txt
  uvicorn main:app --host 0.0.0.0 --port 8000

환경 변수:
  HF_TOKEN   — Hugging Face 토큰 (비공개 모델 또는 빠른 다운로드 시 사용)
  PORT       — 서버 포트 (기본: 8000)
"""

import asyncio
import base64
import hashlib
import json
import os
import re
import secrets
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional
from io import BytesIO
from PIL import Image
import zipfile

from deep_translator import GoogleTranslator

import torch
from diffusers import StableDiffusionPipeline
from transformers import AutoProcessor, AutoModelForCausalLM
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# ── 전역 상태 ─────────────────────────────────────────────────────────────────

pipe: StableDiffusionPipeline | None = None
executor = ThreadPoolExecutor(max_workers=1)  # 순차 생성 (VRAM 충돌 방지)

MODEL_ID = "runwayml/stable-diffusion-v1-5"
HF_TOKEN = os.getenv("HF_TOKEN") or None

# ── 파일 경로 ─────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent
USERS_FILE  = BASE_DIR / "users.json"
TOKENS_FILE = BASE_DIR / "tokens.json"
PAGES_FILE  = BASE_DIR / "pages.json"
GENS_FILE   = BASE_DIR / "generations.json"
STORAGE_DIR = BASE_DIR / "storage"
STORAGE_DIR.mkdir(exist_ok=True)

# ── 메모리 상태 ───────────────────────────────────────────────────────────────

active_tokens: dict[str, str] = {}   # token -> user_id
pages_db:      dict[int, dict] = {}  # page_id -> page
gens_db:       dict[str, dict] = {}  # prompt_id -> generation
jobs:          dict[str, dict] = {}  # prompt_id -> {status, image_paths, error_msg}

# ── 한->영 번역 ────────────────────────────────────────────────────────────────

_KO_PATTERN = re.compile(r'[가-힣ᄀ-ᇿ㄰-㆏]')

def _translate_if_korean(text: str) -> str:
    if not _KO_PATTERN.search(text):
        return text
    try:
        result = GoogleTranslator(source='ko', target='en').translate(text)
        translated = result or text
        print(f"[번역] {text!r} -> {translated!r}")
        return translated
    except Exception as e:
        print(f"[번역] 실패, 원본 사용: {e}")
        return text


# ── Florence-2 (이미지 스타일 추출) ───────────────────────────────────────────

VL_MODEL_ID = "microsoft/Florence-2-base"
_vl_processor: AutoProcessor | None = None
_vl_model:     AutoModelForCausalLM | None = None


def _load_vl() -> tuple[AutoProcessor, AutoModelForCausalLM]:
    global _vl_processor, _vl_model
    if _vl_processor is None:
        import traceback as _tb
        print(f"[VL] 모델 로딩 중: {VL_MODEL_ID}")
        try:
            # Transformers 4.45+ 호환성을 위한 원숭이 패치
            from transformers.configuration_utils import PretrainedConfig
            from transformers import PreTrainedTokenizerFast, RobertaTokenizer
            if not hasattr(PretrainedConfig, "forced_bos_token_id"):
                PretrainedConfig.forced_bos_token_id = None
            if not hasattr(PreTrainedTokenizerFast, "additional_special_tokens"):
                PreTrainedTokenizerFast.additional_special_tokens = property(lambda self: [])
            if not hasattr(RobertaTokenizer, "additional_special_tokens"):
                RobertaTokenizer.additional_special_tokens = property(lambda self: [])

            device = "cuda" if torch.cuda.is_available() else "cpu"
            dtype  = torch.float16 if torch.cuda.is_available() else torch.float32
            print(f"[VL] device={device}, dtype={dtype}")

            _vl_processor = AutoProcessor.from_pretrained(
                VL_MODEL_ID, trust_remote_code=True
            )
            _vl_model = AutoModelForCausalLM.from_pretrained(
                VL_MODEL_ID,
                trust_remote_code=True,
                dtype=dtype,
                attn_implementation="eager",   # transformers 5.x SDPA 호환성 우회
                low_cpu_mem_usage=False,        # meta tensor 비활성화 (Florence-2 구형 코드 호환)
            ).eval().to(device)
            print(f"[VL] ✅ 모델 로드 완료 (device={device})")
        except Exception:
            print("[VL] ❌ 모델 로드 실패:")
            import traceback as _tb
            _tb.print_exc()
            raise
    return _vl_processor, _vl_model


def _extract_style_local(image_b64: str) -> str:
    import traceback as _tb
    try:
        processor, model = _load_vl()
        raw = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
        img = Image.open(BytesIO(base64.b64decode(raw))).convert("RGB")
        task = "<MORE_DETAILED_CAPTION>"
        inputs = processor(text=task, images=img, return_tensors="pt")
        model_dtype = next(model.parameters()).dtype
        # float 텐서(pixel_values 등)만 모델 dtype으로 캐스팅, 정수형(input_ids)은 device만 이동
        inputs = {
            k: (v.to(model.device, dtype=model_dtype) if v.is_floating_point() else v.to(model.device))
            for k, v in inputs.items()
        }
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=200,
                do_sample=False,
            )
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        result = processor.post_process_generation(
            generated_text, task=task, image_size=(img.width, img.height),
        )
        return result[task]
    except Exception as e:
        print("[VL] ❌ 스타일 추출 중 오류:")
        _tb.print_exc()
        raise


# ── 인증 유틸 ─────────────────────────────────────────────────────────────────

def _load_users() -> dict[str, dict]:
    if not USERS_FILE.exists():
        return {}
    return json.loads(USERS_FILE.read_text(encoding="utf-8"))


def _save_users(users: dict[str, dict]) -> None:
    USERS_FILE.write_text(json.dumps(users, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_tokens() -> dict[str, str]:
    if not TOKENS_FILE.exists():
        return {}
    return json.loads(TOKENS_FILE.read_text(encoding="utf-8"))


def _save_tokens() -> None:
    TOKENS_FILE.write_text(json.dumps(active_tokens, ensure_ascii=False, indent=2), encoding="utf-8")


def _hash_pw(password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def require_auth(authorization: str = Header(...)) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="인증 토큰이 필요합니다.")
    token = authorization[7:]
    user_id = active_tokens.get(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="유효하지 않거나 만료된 토큰입니다.")
    return user_id


# ── Pages / Generations 영속화 ────────────────────────────────────────────────

def _load_pages() -> dict[int, dict]:
    if not PAGES_FILE.exists():
        return {}
    raw = json.loads(PAGES_FILE.read_text(encoding="utf-8"))
    return {int(k): v for k, v in raw.items()}


def _save_pages() -> None:
    PAGES_FILE.write_text(json.dumps(pages_db, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_gens() -> dict[str, dict]:
    if not GENS_FILE.exists():
        return {}
    return json.loads(GENS_FILE.read_text(encoding="utf-8"))


def _save_gens() -> None:
    GENS_FILE.write_text(json.dumps(gens_db, ensure_ascii=False, indent=2), encoding="utf-8")


def _next_page_id() -> int:
    return max(pages_db.keys(), default=0) + 1


# ── SD 이미지 생성 ─────────────────────────────────────────────────────────────

def _generate_one(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
) -> str:
    """Stable Diffusion 1.5로 이미지 1장 생성 → base64 data URI 반환"""
    assert pipe is not None, "모델이 로드되지 않았습니다."
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
    )
    img = result.images[0]
    buf = BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _run_generation_job(prompt_id: str, prompt: str, count: int, page_id: int) -> None:
    """ThreadPoolExecutor에서 실행되는 동기 Job 함수"""
    try:
        jobs[prompt_id]["status"] = "running"
        negative_prompt = "blurry, low quality, ugly, deformed, watermark"
        image_paths = []

        for i in range(count):
            data_uri = _generate_one(prompt, negative_prompt, 512, 512, 20, 7.5)
            # base64 → 파일 저장
            raw = data_uri.split(",", 1)[1]
            img_bytes = base64.b64decode(raw)
            rel_path = f"{page_id}/{prompt_id}_{i}.png"
            out_path = STORAGE_DIR / str(page_id) / f"{prompt_id}_{i}.png"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(img_bytes)
            image_paths.append(rel_path)
            print(f"[SD] {prompt_id} — {i+1}/{count} 완료")

        jobs[prompt_id]["status"] = "done"
        jobs[prompt_id]["image_paths"] = image_paths

        # generations_db 업데이트
        if prompt_id in gens_db:
            gens_db[prompt_id]["status"] = "done"
            gens_db[prompt_id]["image_paths"] = image_paths
            _save_gens()

    except Exception as e:
        print(f"[SD] {prompt_id} 생성 실패: {e}")
        jobs[prompt_id]["status"] = "error"
        jobs[prompt_id]["error_msg"] = str(e)
        if prompt_id in gens_db:
            gens_db[prompt_id]["status"] = "error"
            gens_db[prompt_id]["error_msg"] = str(e)
            _save_gens()


# ── 앱 생명주기 ───────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe, active_tokens, pages_db, gens_db

    # 저장 데이터 복원
    active_tokens = _load_tokens()
    pages_db = _load_pages()
    gens_db = _load_gens()
    print(f"[인증] 저장된 토큰 {len(active_tokens)}개 복원")
    print(f"[DB]  페이지 {len(pages_db)}개, 생성 {len(gens_db)}개 복원")

    # ── GPU / CPU 감지 로그 ────────────────────────────────────────────────────
    print("=" * 60)
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"[GPU] ✅ CUDA GPU 감지됨 — 총 {gpu_count}개")
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / 1024 ** 3
            print(f"[GPU] [{i}] {props.name}")
            print(f"[GPU]     VRAM        : {vram_gb:.1f} GB")
            print(f"[GPU]     CUDA Cores  : {props.multi_processor_count} SMs")
            print(f"[GPU]     Compute Cap : {props.major}.{props.minor}")
        print(f"[GPU] CUDA 버전   : {torch.version.cuda}")
        print(f"[GPU] PyTorch     : {torch.__version__}")
        print(f"[GPU] 사용 dtype  : float16 (FP16 가속)")
    else:
        import platform, subprocess
        cpu_name = "알 수 없음"
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["powershell", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                    capture_output=True, text=True, timeout=5
                )
                cpu_name = result.stdout.strip() or "알 수 없음"
            else:
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if "model name" in line:
                            cpu_name = line.split(":")[1].strip()
                            break
        except Exception:
            pass
        print("[GPU] ❌ CUDA GPU 없음 — CPU 모드로 실행합니다")
        print(f"[CPU] {cpu_name}")
        print(f"[CPU] PyTorch     : {torch.__version__}")
        print(f"[CPU] 사용 dtype  : float32 (FP32)")
        print("[CPU] ⚠️  이미지 생성이 매우 느릴 수 있습니다 (GPU 권장)")
    print("=" * 60)

    # SD 모델 로드
    print(f"[SD] 모델 로딩 중: {MODEL_ID}")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            token=HF_TOKEN,
            safety_checker=None,
            requires_safety_checker=False,
        )
        if torch.cuda.is_available():
            pipe.to("cuda")
            pipe.enable_attention_slicing()
            print(f"[SD] GPU 로드 완료 ✅ ({torch.cuda.get_device_name(0)})")
        else:
            pipe.to("cpu")
            print("[SD] CPU 로드 완료 ✅")
        print("[SD] 모델 준비 완료 — 이미지 생성 요청을 받을 수 있습니다 🎨")
    except Exception as e:
        print(f"[SD] 모델 로드 실패 ❌: {e}")
    yield
    del pipe


# ── FastAPI 앱 ────────────────────────────────────────────────────────────────

app = FastAPI(title="Batch Image Studio API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 생성 이미지 정적 파일 서빙
app.mount("/storage", StaticFiles(directory=str(STORAGE_DIR)), name="storage")


# ── 인증 스키마 ───────────────────────────────────────────────────────────────

class AuthRequest(BaseModel):
    username: str
    password: str

class AuthResponse(BaseModel):
    access_token: str
    user_id: str
    username: str


# ╔══════════════════════════════════════════════════════════════════╗
# ║                   /api/v1 라우터                                 ║
# ╚══════════════════════════════════════════════════════════════════╝

# ── 인증 ─────────────────────────────────────────────────────────────────────

@app.post("/api/v1/auth/register", response_model=AuthResponse)
async def api_register(req: AuthRequest):
    users = _load_users()
    # username 중복 확인
    if any(u["username"] == req.username for u in users.values()):
        raise HTTPException(status_code=409, detail="이미 사용 중인 사용자명입니다.")
    user_id = secrets.token_hex(8)
    salt = secrets.token_hex(16)
    users[user_id] = {
        "id": user_id,
        "username": req.username,
        "salt": salt,
        "password_hash": _hash_pw(req.password, salt),
    }
    _save_users(users)
    token = secrets.token_hex(32)
    active_tokens[token] = user_id
    _save_tokens()
    return AuthResponse(access_token=token, user_id=user_id, username=req.username)


@app.post("/api/v1/auth/login", response_model=AuthResponse)
async def api_login(req: AuthRequest):
    users = _load_users()
    user = next((u for u in users.values() if u["username"] == req.username), None)
    if not user or _hash_pw(req.password, user["salt"]) != user["password_hash"]:
        raise HTTPException(status_code=401, detail="사용자명 또는 비밀번호가 올바르지 않습니다.")
    token = secrets.token_hex(32)
    active_tokens[token] = user["id"]
    _save_tokens()
    return AuthResponse(access_token=token, user_id=user["id"], username=user["username"])


@app.post("/api/v1/auth/logout")
async def api_logout(authorization: str = Header(...)):
    if authorization.startswith("Bearer "):
        active_tokens.pop(authorization[7:], None)
        _save_tokens()
    return {"ok": True}


@app.get("/api/v1/auth/me")
async def api_me(user_id: str = Depends(require_auth)):
    users = _load_users()
    user = users.get(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="사용자를 찾을 수 없습니다.")
    return {"user_id": user["id"], "username": user["username"]}


# ── Pages ─────────────────────────────────────────────────────────────────────

class PageCreateRequest(BaseModel):
    title: str = "새 채팅"

class PagePatchRequest(BaseModel):
    title: Optional[str] = None


@app.get("/api/v1/pages")
async def api_list_pages(user_id: str = Depends(require_auth)):
    user_pages = [p for p in pages_db.values() if p["user_id"] == user_id]
    user_pages.sort(key=lambda p: p["created_at"], reverse=True)
    return [{"id": p["id"], "title": p["title"], "created_at": p["created_at"]} for p in user_pages]


@app.post("/api/v1/pages")
async def api_create_page(req: PageCreateRequest, user_id: str = Depends(require_auth)):
    page_id = _next_page_id()
    page = {
        "id": page_id,
        "user_id": user_id,
        "title": req.title,
        "created_at": datetime.now().isoformat(),
    }
    pages_db[page_id] = page
    _save_pages()
    return {"id": page_id, "title": page["title"], "created_at": page["created_at"]}


@app.patch("/api/v1/pages/{page_id}")
async def api_update_page(page_id: int, req: PagePatchRequest, user_id: str = Depends(require_auth)):
    page = pages_db.get(page_id)
    if not page or page["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="페이지를 찾을 수 없습니다.")
    if req.title is not None:
        pages_db[page_id]["title"] = req.title
    _save_pages()
    return {"id": page_id, "title": pages_db[page_id]["title"]}


@app.delete("/api/v1/pages/{page_id}")
async def api_delete_page(page_id: int, user_id: str = Depends(require_auth)):
    page = pages_db.get(page_id)
    if not page or page["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="페이지를 찾을 수 없습니다.")
    pages_db.pop(page_id, None)
    _save_pages()
    return {"ok": True}


# ── Generations ───────────────────────────────────────────────────────────────

@app.get("/api/v1/pages/{page_id}/generations")
async def api_get_generations(page_id: int, user_id: str = Depends(require_auth)):
    page = pages_db.get(page_id)
    if not page or page["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="페이지를 찾을 수 없습니다.")
    page_gens = [g for g in gens_db.values() if g.get("page_id") == page_id]
    page_gens.sort(key=lambda g: g.get("created_at", 0))
    return page_gens


class GenerateRequest(BaseModel):
    prompt: str
    id: Optional[str] = None
    count: int = Field(default=1, ge=1, le=8)
    page_id: Optional[int] = None


@app.post("/api/v1/generate", status_code=202)
async def api_generate(req: GenerateRequest, user_id: str = Depends(require_auth)):
    if pipe is None:
        raise HTTPException(status_code=503, detail="모델 로딩 중입니다. /health 에서 model_loaded가 true가 될 때까지 대기하세요.")

    prompt_id = req.id or secrets.token_hex(8)
    page_id = req.page_id

    # page_id가 없으면 새 페이지 자동 생성
    if page_id is None:
        new_page_id = _next_page_id()
        pages_db[new_page_id] = {
            "id": new_page_id,
            "user_id": user_id,
            "title": req.prompt[:30],
            "created_at": datetime.now().isoformat(),
        }
        _save_pages()
        page_id = new_page_id

    # Job 초기화
    jobs[prompt_id] = {"status": "pending", "image_paths": [], "error_msg": None}

    # Generations DB에 등록
    gens_db[prompt_id] = {
        "prompt_id": prompt_id,
        "page_id": page_id,
        "prompt_text": req.prompt,
        "status": "pending",
        "image_paths": [],
        "error_msg": None,
        "created_at": datetime.now().isoformat(),
    }
    _save_gens()

    # 번역 후 비동기 실행
    loop = asyncio.get_event_loop()
    translated = await loop.run_in_executor(None, _translate_if_korean, req.prompt)
    loop.run_in_executor(executor, _run_generation_job, prompt_id, translated, req.count, page_id)

    return {"prompt_id": prompt_id, "page_id": page_id, "status": "pending"}


@app.get("/api/v1/generations/{prompt_id}/status")
async def api_generation_status(prompt_id: str):
    job = jobs.get(prompt_id)
    if job is None:
        # DB에서 복원 시도
        gen = gens_db.get(prompt_id)
        if gen:
            return {
                "status": gen["status"],
                "image_paths": gen.get("image_paths", []),
                "error_msg": gen.get("error_msg"),
            }
        raise HTTPException(status_code=404, detail="해당 작업을 찾을 수 없습니다.")
    return {
        "status": job["status"],
        "image_paths": job.get("image_paths", []),
        "error_msg": job.get("error_msg"),
    }


# ── 스타일 추출 ───────────────────────────────────────────────────────────────

class ExtractStyleRequest(BaseModel):
    image: str  # base64 data URI


@app.post("/api/v1/extract-style")
async def api_extract_style(req: ExtractStyleRequest, _: str = Depends(require_auth)):
    import traceback as _tb
    try:
        loop = asyncio.get_event_loop()
        style = await loop.run_in_executor(None, _extract_style_local, req.image)
        return {"style": style}
    except Exception as e:
        print(f"[VL] API 에러: {e}")
        _tb.print_exc()
        raise HTTPException(status_code=500, detail=f"스타일 추출 실패: {e}")


# ── ZIP 다운로드 ───────────────────────────────────────────────────────────────

@app.get("/api/v1/pages/{page_id}/download-zip")
async def api_download_zip(page_id: int, user_id: str = Depends(require_auth)):
    page = pages_db.get(page_id)
    if not page or page["user_id"] != user_id:
        raise HTTPException(status_code=404, detail="페이지를 찾을 수 없습니다.")

    page_gens = [g for g in gens_db.values() if g.get("page_id") == page_id and g.get("image_paths")]

    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for gen in page_gens:
            prompt_id = gen["prompt_id"]
            for i, rel_path in enumerate(gen.get("image_paths", [])):
                file_path = STORAGE_DIR / rel_path
                if file_path.exists():
                    zf.write(file_path, arcname=f"{prompt_id}/{i}.png")
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename=carbatch-page-{page_id}.zip"},
    )


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model_loaded": pipe is not None,
        "model_id": MODEL_ID,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


# ── (하위 호환) 레거시 엔드포인트 ─────────────────────────────────────────────

class LegacyRegisterRequest(BaseModel):
    username: str = Field(min_length=2, max_length=20)
    email: str = Field(min_length=5)
    password: str = Field(min_length=6)

class LegacyLoginRequest(BaseModel):
    email: str
    password: str

@app.post("/auth/register")
async def legacy_register(req: LegacyRegisterRequest):
    return await api_register(AuthRequest(username=req.username, password=req.password))

@app.post("/auth/login")
async def legacy_login(req: LegacyLoginRequest):
    users = _load_users()
    user = next((u for u in users.values() if u.get("email") == req.email), None)
    if not user or _hash_pw(req.password, user["salt"]) != user["password_hash"]:
        raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 올바르지 않습니다.")
    token = secrets.token_hex(32)
    active_tokens[token] = user["id"]
    _save_tokens()
    return {"token": token, "user_id": user["id"], "username": user["username"]}
