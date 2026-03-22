"""
BE/main.py

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
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from io import BytesIO

from deep_translator import GoogleTranslator

import torch
from diffusers import StableDiffusionPipeline
from transformers import BlipProcessor, BlipForConditionalGeneration
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── 전역 상태 ─────────────────────────────────────────────────────────────────

pipe: StableDiffusionPipeline | None = None
executor = ThreadPoolExecutor(max_workers=1)  # 순차 생성 (VRAM 충돌 방지)

MODEL_ID = "runwayml/stable-diffusion-v1-5"
HF_TOKEN = os.getenv("HF_TOKEN") or None

# ── 인증 — 사용자 저장소 ──────────────────────────────────────────────────────


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


# ── BLIP 이미지 캡셔닝 (lazy load) ──────────────────────────────────────────

_blip_processor: BlipProcessor | None = None
_blip_model: BlipForConditionalGeneration | None = None


def _load_blip() -> tuple[BlipProcessor, BlipForConditionalGeneration]:
    global _blip_processor, _blip_model
    if _blip_processor is None:
        print("[BLIP] 모델 로딩 중...")
        _blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
        _blip_model.eval()
        print("[BLIP] 모델 로드 완료")
    return _blip_processor, _blip_model


def _caption_image(image_b64: str) -> str:
    proc, model = _load_blip()
    raw = image_b64.split(",", 1)[1] if "," in image_b64 else image_b64
    img = Image.open(BytesIO(base64.b64decode(raw))).convert("RGB")
    inputs = proc(img, return_tensors="pt")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=120)
    return proc.decode(out[0], skip_special_tokens=True)


USERS_FILE  = Path(__file__).parent / "users.json"
TOKENS_FILE = Path(__file__).parent / "tokens.json"
active_tokens: dict[str, str] = {}  # token -> user_id


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
    """Authorization: Bearer <token> 헤더에서 user_id 추출"""
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="인증 토큰이 필요합니다.")
    token = authorization[7:]
    user_id = active_tokens.get(token)
    if not user_id:
        raise HTTPException(status_code=401, detail="유효하지 않거나 만료된 토큰입니다.")
    return user_id


# ── 앱 생명주기 (시작 시 모델 로드) ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe, active_tokens
    active_tokens = _load_tokens()
    print(f"[인증] 저장된 토큰 {len(active_tokens)}개 복원")
    print(f"[SD] 모델 로딩 중: {MODEL_ID}")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            token=HF_TOKEN,
            safety_checker=None,          # NSFW 필터 비활성화 (속도 향상)
            requires_safety_checker=False,
        )
        if torch.cuda.is_available():
            pipe.to("cuda")
            pipe.enable_attention_slicing()   # VRAM 절약
            print(f"[SD] GPU 사용 ({torch.cuda.get_device_name(0)})")
        else:
            pipe.to("cpu")
            print("[SD] CPU 모드 (생성 속도가 느릴 수 있음)")
        print("[SD] 모델 로드 완료 ✓")
    except Exception as e:
        print(f"[SD] 모델 로드 실패: {e}")
        # 서버는 계속 실행하되 /health 에서 상태를 알림
    yield
    # 종료 정리
    del pipe


# ── FastAPI 앱 ────────────────────────────────────────────────────────────────

app = FastAPI(title="Stable Diffusion 1.5 API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 인증 스키마 ───────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(min_length=2, max_length=20)
    email: str = Field(min_length=5)
    password: str = Field(min_length=6)

class LoginRequest(BaseModel):
    email: str
    password: str

class AuthResponse(BaseModel):
    token: str
    user_id: str
    username: str
    email: str


# ── 인증 엔드포인트 ────────────────────────────────────────────────────────────

@app.post("/auth/register", response_model=AuthResponse)
async def register(req: RegisterRequest):
    users = _load_users()
    if any(u["email"] == req.email for u in users.values()):
        raise HTTPException(status_code=409, detail="이미 사용 중인 이메일입니다.")
    user_id = secrets.token_hex(8)
    salt = secrets.token_hex(16)
    users[user_id] = {
        "id": user_id,
        "username": req.username,
        "email": req.email,
        "salt": salt,
        "password_hash": _hash_pw(req.password, salt),
    }
    _save_users(users)
    token = secrets.token_hex(32)
    active_tokens[token] = user_id
    _save_tokens()
    return AuthResponse(token=token, user_id=user_id, username=req.username, email=req.email)


@app.post("/auth/login", response_model=AuthResponse)
async def login(req: LoginRequest):
    users = _load_users()
    user = next((u for u in users.values() if u["email"] == req.email), None)
    if not user or _hash_pw(req.password, user["salt"]) != user["password_hash"]:
        raise HTTPException(status_code=401, detail="이메일 또는 비밀번호가 올바르지 않습니다.")
    token = secrets.token_hex(32)
    active_tokens[token] = user["id"]
    return AuthResponse(token=token, user_id=user["id"], username=user["username"], email=user["email"])


@app.post("/auth/logout")
async def logout(authorization: str = Header(...)):
    if authorization.startswith("Bearer "):
        active_tokens.pop(authorization[7:], None)
        _save_tokens()
    return {"ok": True}


# ── 요청 / 응답 스키마 ────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    model: Literal["sd15", "sd15-lcm"] = "sd15"  # sd15-lcm은 향후 구현 예정
    prompt: str
    negative_prompt: str = Field(default="blurry, low quality, ugly, deformed, watermark")
    count: int = Field(default=1, ge=1, le=8)
    width: int = Field(default=512, ge=256, le=1024)
    height: int = Field(default=512, ge=256, le=1024)
    num_inference_steps: int = Field(default=20, ge=1, le=100)
    guidance_scale: float = Field(default=7.5, ge=1.0, le=20.0)


class GenerateResponse(BaseModel):
    success: bool
    images: list[str | None]
    error: str | None = None


# ── 동기 생성 헬퍼 (ThreadPoolExecutor에서 실행) ──────────────────────────────

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


# ── 엔드포인트 ────────────────────────────────────────────────────────────────

class AnalyzeImageRequest(BaseModel):
    image: str  # base64 data URI


@app.post("/analyze-image")
async def analyze_image(req: AnalyzeImageRequest, _: str = Depends(require_auth)):
    """이미지에서 스타일 프롬프트 추출 (BLIP 캡셔닝)"""
    try:
        loop = asyncio.get_event_loop()
        caption = await loop.run_in_executor(None, _caption_image, req.image)
        return {"style_prompt": caption}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 분석 실패: {e}")


@app.get("/health")
async def health():
    """서버 및 모델 상태 확인"""
    return {
        "status": "ok",
        "model_loaded": pipe is not None,
        "model_id": MODEL_ID,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest, _: str = Depends(require_auth)):
    """프롬프트로 이미지를 count장 생성합니다."""
    if pipe is None:
        raise HTTPException(
            status_code=503,
            detail="모델 로딩 중입니다. /health 에서 model_loaded 가 true가 될 때까지 대기하세요.",
        )

    if req.model == "sd15-lcm":
        print("[SD] sd15-lcm 요청 수신 — 현재 SD 1.5로 폴백 (LCM 미구현)")

    prompt = await asyncio.get_event_loop().run_in_executor(None, _translate_if_korean, req.prompt)

    loop = asyncio.get_event_loop()
    images: list[str | None] = []
    first_error: str | None = None

    for i in range(req.count):
        try:
            img = await loop.run_in_executor(
                executor,
                lambda: _generate_one(
                    prompt,
                    req.negative_prompt,
                    req.width,
                    req.height,
                    req.num_inference_steps,
                    req.guidance_scale,
                ),
            )
            images.append(img)
            print(f"[SD] 생성 완료 {i + 1}/{req.count}")
        except Exception as e:
            images.append(None)
            if first_error is None:
                first_error = str(e)
            print(f"[SD] 생성 실패 {i + 1}/{req.count}: {e}")

    success = any(img is not None for img in images)
    return GenerateResponse(
        success=success,
        images=images,
        error=first_error if not success else None,
    )
