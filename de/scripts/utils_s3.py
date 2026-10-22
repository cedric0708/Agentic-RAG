"""
S3 공통 유틸리티
모든 수집/ETL 스크립트에서 import해서 사용
"""
import os
import json
import gzip
from datetime import datetime
from typing import Any

import boto3
from botocore.exceptions import ClientError
from loguru import logger

# config/.env를 여기서 먼저 읽음 → 이 파일을 import하는 모든 수집/ETL 스크립트에 환경변수 적용
# (이미 설정된 환경변수는 덮어쓰지 않음)
try:
    from pathlib import Path
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / "config" / ".env", override=False)
except ImportError:
    pass


def run_date() -> str:
    """bronze/silver 날짜 폴더 이름.
    .env 또는 환경변수에 RUN_DATE=YYYY-MM-DD가 있으면 그 날짜로 고정 (여러 날에 걸쳐 수집해도 한 폴더에 이어 저장)
    없으면 오늘 날짜"""
    return os.getenv("RUN_DATE") or datetime.now().strftime("%Y-%m-%d")

def get_s3_client():
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

BUCKET = os.getenv("S3_BUCKET")


def bronze_key(source: str, filename: str) -> str:
    """bronze zone 경로 생성 - 날짜 파티셔닝 포함 (RUN_DATE 우선)"""
    return f"bronze/{source}/{run_date()}/{filename}"


def upload_json(data: Any, s3_key: str) -> bool:
    """딕셔너리/리스트를 JSON으로 S3에 업로드"""
    try:
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        get_s3_client().put_object(
            Bucket=BUCKET,
            Key=s3_key,
            Body=body,
            ContentType="application/json; charset=utf-8",
        )
        logger.info(f"✅ 업로드 완료: s3://{BUCKET}/{s3_key}")
        return True
    except ClientError as e:
        logger.error(f"❌ 업로드 실패 [{s3_key}]: {e}")
        return False


def key_exists(s3_key: str) -> bool:
    """S3 키 존재 여부 확인 (중복 수집 방지)"""
    try:
        get_s3_client().head_object(Bucket=BUCKET, Key=s3_key)
        return True
    except ClientError:
        return False


def list_keys(prefix: str) -> list:
    """prefix 하위 키 목록 반환"""
    paginator = get_s3_client().get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


DATE_FOLDER_RE = __import__("re").compile(r"/(\d{4}-\d{2}-\d{2})/")


def dated_keys(prefix: str, suffix: str = ".json") -> list:
    """prefix 아래 파일 중 '원천 폴더별로 한 날짜'의 파일만 반환
    - 원천 폴더(날짜 앞까지 경로)마다 RUN_DATE 폴더가 있으면 그 날짜, 없으면 가장 최신 날짜
    - 예전 날짜 데이터가 섞이거나, 여러 날 수집분이 중복으로 읽히는 것을 막음"""
    groups: dict = {}
    for k in list_keys(prefix):
        if not k.endswith(suffix):
            continue
        m = DATE_FOLDER_RE.search(k)
        if not m:
            continue
        groups.setdefault(k[:m.start()], {}).setdefault(m.group(1), []).append(k)
    rd = os.getenv("RUN_DATE")
    out = []
    for base, by_date in sorted(groups.items()):
        date = rd if rd in by_date else max(by_date)
        if rd and date != rd:
            logger.warning(f"  {base}/ 에 RUN_DATE({rd}) 폴더가 없어 최신 {date} 사용")
        if len(by_date) > 1:
            logger.info(f"  {base}/ 날짜 {len(by_date)}개 중 {date} 사용 ({len(by_date[date])}개 파일)")
        out += by_date[date]
    return sorted(out)


def download_json(s3_key: str) -> Any:
    """S3에서 JSON 다운로드"""
    try:
        response = get_s3_client().get_object(Bucket=BUCKET, Key=s3_key)
        return json.loads(response["Body"].read().decode("utf-8"))
    except ClientError as e:
        logger.error(f"❌ 다운로드 실패 [{s3_key}]: {e}")
        return None


def silver_key(category: str, filename: str) -> str:
    """silver zone S3 키 생성 (RUN_DATE 우선)"""
    return f"silver/{category}/{run_date()}/{filename}"
