# BE Codebase Reference

## 실행 방법

```bash
# venv 활성화
venv\Scripts\activate

# 서버 실행
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 환경 변수 (`.env` 또는 PowerShell `$env:`)

| 변수 | 기본값 | 설명 |
|---|---|---|
| `HF_TOKEN` | None | HuggingFace 토큰 (캐시 가속) |
| `PORT` | 8000 | 서버 포트 (현재 uvicorn 인자로 지정) |

---

## AI 모델

### 1. Stable Diffusion 1.5 — 이미지 생성
- **모델 ID**: `runwayml/stable-diffusion-v1-5`
- **전역 변수**: `pipe` (`StableDiffusionPipeline | None`)
- **로드 시점**: 서버 시작 시 `lifespan()` 에서 자동 로드
- **디바이스**: GPU(CUDA) 우선, 없으면 CPU
- **VRAM 최적화**: `float16` + `enable_attention_slicing()`
- **safety_checker**: 비활성화 (속도 향상)
- **캐시 위치**: `C:\Users\PC\.cache\huggingface\hub\`
- **용량**: ~4GB

### 2. Florence-2-base — 이미지 스타일 추출
- **모델 ID**: `microsoft/Florence-2-base`
- **전역 변수**: `_vl_processor`, `_vl_model`
- **로드 시점**: `/analyze-image` 첫 호출 시 lazy load
- **디바이스**: GPU 우선, 없으면 CPU (`float16` / `float32`)
- **사용 태스크**: `<MORE_DETAILED_CAPTION>`
- **용량**: ~460MB
- **주의**: `trust_remote_code=True` 필요

### 3. deep-translator — 한→영 번역
- **라이브러리**: `deep-translator` (GoogleTranslator)
- **동작**: 프롬프트에 한글 포함 시 자동 번역 후 SD에 전달
- **패턴**: `_KO_PATTERN = re.compile(r'[가-힣ᄀ-ᇿ㄰-㆏]')`

---

## 파일 구조

```
BE/
├── main.py          # FastAPI 서버 (전체 로직)
├── requirements.txt # 패키지 목록
├── CODEBASE.md      # 이 파일
├── users.json       # 사용자 DB (자동 생성, git 제외)
├── tokens.json      # 세션 토큰 DB (자동 생성, git 제외)
└── venv/            # 가상환경 (git 제외)
```

---

## 전역 상태

```python
pipe: StableDiffusionPipeline | None        # SD 1.5 파이프라인
executor = ThreadPoolExecutor(max_workers=1) # SD 순차 처리용 (VRAM 충돌 방지)
_vl_processor: AutoProcessor | None         # Florence-2 프로세서
_vl_model: AutoModelForCausalLM | None      # Florence-2 모델
active_tokens: dict[str, str]               # token -> user_id (메모리 + tokens.json 영속)
```

---

## 함수 목록

### 번역
| 함수 | 설명 |
|---|---|
| `_translate_if_korean(text)` | 한글 감지 후 영어 번역. 영어면 그대로 반환 |

### Florence-2 (이미지 분석)
| 함수 | 설명 |
|---|---|
| `_load_vl()` | Florence-2 lazy load. 이미 로드됐으면 캐시 반환 |
| `_extract_style_local(image_b64)` | base64 이미지 → 스타일 텍스트 추출 |

### 인증
| 함수 | 설명 |
|---|---|
| `_load_users()` | `users.json` 읽기 |
| `_save_users(users)` | `users.json` 쓰기 |
| `_load_tokens()` | `tokens.json` 읽기 |
| `_save_tokens()` | `tokens.json` 쓰기 (`active_tokens` 전체 저장) |
| `_hash_pw(password, salt)` | SHA-256 해시 |
| `require_auth(authorization)` | FastAPI Depends. Bearer 토큰 검증 → user_id 반환 |

### SD 생성
| 함수 | 설명 |
|---|---|
| `_generate_one(prompt, negative_prompt, width, height, steps, guidance_scale)` | SD 1.5로 이미지 1장 생성 → base64 data URI 반환. 동기 함수, executor에서 실행 |

### 앱 생명주기
| 함수 | 설명 |
|---|---|
| `lifespan(app)` | 서버 시작 시 tokens 복원 + SD 모델 로드. 종료 시 pipe 삭제 |

---

## API 엔드포인트

### 인증 (auth 불필요)

| 메서드 | 경로 | 설명 |
|---|---|---|
| `POST` | `/auth/register` | 회원가입. `username(2~20자)`, `email`, `password(6자+)` |
| `POST` | `/auth/login` | 로그인. `email`, `password` |
| `POST` | `/auth/logout` | 로그아웃. Bearer 토큰 무효화 + tokens.json 저장 |
| `GET` | `/health` | 서버/모델 상태 확인 |

### 기능 (Bearer 토큰 필요 — `require_auth`)

| 메서드 | 경로 | 설명 |
|---|---|---|
| `POST` | `/generate` | 이미지 생성 (SD 1.5) |
| `POST` | `/analyze-image` | 이미지 → 스타일 텍스트 (Florence-2) |

---

## 요청/응답 스키마

### `POST /auth/register`, `/auth/login` → `AuthResponse`
```json
// 요청
{ "username": "홍길동", "email": "a@b.com", "password": "123456" }

// 응답
{ "token": "abc...", "user_id": "a1b2c3d4", "username": "홍길동", "email": "a@b.com" }
```

### `POST /generate` → `GenerateResponse`
```json
// 요청
{
  "model": "sd15",          // "sd15" | "sd15-lcm" (lcm은 현재 sd15로 폴백)
  "prompt": "a cat",
  "negative_prompt": "blurry, low quality...",
  "count": 2,               // 1~8
  "width": 512,             // 256~1024
  "height": 512,            // 256~1024
  "num_inference_steps": 20,
  "guidance_scale": 7.5
}

// 응답
{
  "success": true,
  "images": ["data:image/png;base64,...", null],
  "error": null
}
```

### `POST /analyze-image` → `{ style_prompt: string }`
```json
// 요청
{ "image": "data:image/png;base64,..." }

// 응답
{ "style_prompt": "soft natural lighting, warm color palette..." }
```

### `GET /health`
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_id": "runwayml/stable-diffusion-v1-5",
  "device": "cuda"
}
```

---

## 사용자 데이터 구조 (users.json)

```json
{
  "a1b2c3d4": {
    "id": "a1b2c3d4",
    "username": "홍길동",
    "email": "a@b.com",
    "salt": "랜덤16바이트hex",
    "password_hash": "SHA256(salt+password)"
  }
}
```

## 토큰 데이터 구조 (tokens.json)

```json
{
  "토큰hex64자": "user_id"
}
```
> 서버 재시작 시 `tokens.json` 복원 → 로그인 유지됨

---

## 주요 설계 결정

| 항목 | 결정 | 이유 |
|---|---|---|
| SD 동시성 | `max_workers=1` | VRAM 충돌 방지 |
| Florence-2 로드 | lazy load | 서버 시작 속도 / 미사용 시 메모리 절약 |
| 토큰 저장 | `tokens.json` 파일 | 서버 재시작 후에도 로그인 유지 |
| 비밀번호 | SHA-256 + salt | DB 없이 파일 기반 구현 |
| 번역 | `deep-translator` (Google) | 무료, 설치 간단 |
| CORS | `allow_origins=["*"]` | 개발 환경. 프로덕션 시 제한 필요 |

---

## 향후 미구현 / 개발 예정

- `sd15-lcm` 모델 지원 (현재 sd15로 폴백)
- 서버 스토리지 (Pro/Business 요금제용 이미지 저장)
- 서버 스토리지 용량 제한 (40GB / 100GB)
- 요금제별 기능 분기 로직