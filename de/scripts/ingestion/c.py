import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
import requests
from loguru import logger
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

_klen = lambda: len(os.getenv("PUBLIC_API_KEY") or "")

ENV_PATH = Path(__file__).resolve().parent.parent.parent / "config" / ".env"
print("[0] 실행 파일   :", Path(__file__).resolve())
print("[0] .env 경로   :", ENV_PATH, "| 존재:", ENV_PATH.exists())
print("[1] 로드 결과   :", load_dotenv(ENV_PATH), "| 키 길이:", _klen())

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils_s3 import bronze_key, upload_json, key_exists
print("[2] utils_s3 후 :", "키 길이:", _klen())

API_KEY = os.getenv("PUBLIC_API_KEY")
print("[3] API_KEY     :", "길이:", len(API_KEY or ""))