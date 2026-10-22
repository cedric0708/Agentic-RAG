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

def get_s3_client():
    return boto3.client(
        "s3",
        region_name=os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2"),
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )

BUCKET = os.getenv("S3_BUCKET")


def bronze_key(source: str, filename: str) -> str:
    """bronze zone 경로 생성 - 날짜 파티셔닝 포함"""
    date = datetime.now().strftime("%Y-%m-%d")
    return f"bronze/{source}/{date}/{filename}"


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


def download_json(s3_key: str) -> Any:
    """S3에서 JSON 다운로드"""
    try:
        response = get_s3_client().get_object(Bucket=BUCKET, Key=s3_key)
        return json.loads(response["Body"].read().decode("utf-8"))
    except ClientError as e:
        logger.error(f"❌ 다운로드 실패 [{s3_key}]: {e}")
        return None


def silver_key(category: str, filename: str) -> str:
    """silver zone S3 키 생성"""
    date = datetime.now().strftime("%Y-%m-%d")
    return f"silver/{category}/{date}/{filename}"
