# medicine-backend/main.py
# 실행방법

# 콘다 가상환경 실행 conda activate medi-env

# 서버 실행 uvicorn main:app --reload

import os
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / '.env')
# 로컬 기본 실행에는 Docker/Chroma 서버가 필요하지 않습니다.
os.environ.setdefault('USE_CHROMA', 'false')

from api import diagnosis, user, medicines, pharmacies, interaction_graph
from agents.loader import load_drugs, load_symptoms

app = FastAPI(title="의약품 통합 헬스케어 서버")

# CORS 허용 (React 포트 통신용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# RAG용 마스터 데이터를 서버 시작 시 로드
DATA_DIR = Path(os.getenv('DATA_DIR', PROJECT_ROOT / 'data'))
app.state.DRUGS = load_drugs(DATA_DIR / "drugs.json")
app.state.SYMPTOMS = load_symptoms(DATA_DIR / "symptoms.json")


app.include_router(diagnosis.router)
app.include_router(user.router)
app.include_router(medicines.router)
app.include_router(pharmacies.router)
app.include_router(interaction_graph.router)

@app.get("/")
def read_root():
    return {"status": "running", "architecture": "Agentic-RAG Layered Architecture"}
