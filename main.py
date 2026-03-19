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
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from io import BytesIO

import torch
from diffusers import StableDiffusionPipeline
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── 전역 상태 ─────────────────────────────────────────────────────────────────

pipe: StableDiffusionPipeline | None = None
executor = ThreadPoolExecutor(max_workers=1)  # 순차 생성 (VRAM 충돌 방지)

MODEL_ID = "runwayml/stable-diffusion-v1-5"
HF_TOKEN = os.getenv("HF_TOKEN") or None


# ── 앱 생명주기 (시작 시 모델 로드) ──────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipe
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
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 요청 / 응답 스키마 ────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
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
async def generate(req: GenerateRequest):
    """프롬프트로 이미지를 count장 생성합니다."""
    if pipe is None:
        raise HTTPException(
            status_code=503,
            detail="모델 로딩 중입니다. /health 에서 model_loaded 가 true가 될 때까지 대기하세요.",
        )

    loop = asyncio.get_event_loop()
    images: list[str | None] = []
    first_error: str | None = None

    for i in range(req.count):
        try:
            img = await loop.run_in_executor(
                executor,
                lambda: _generate_one(
                    req.prompt,
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
