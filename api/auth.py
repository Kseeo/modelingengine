"""API 키 검증 (X-API-Key 헤더)."""
from __future__ import annotations

import logging
import os
import secrets

from fastapi import Header, HTTPException

log = logging.getLogger(__name__)

API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    API_KEY = secrets.token_urlsafe(24)
    log.warning(
        "API_KEY 환경변수가 설정되지 않아 임시 키를 생성했습니다 (서버 재시작 시 바뀜): %s", API_KEY
    )


async def verify_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> None:
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(401, "유효하지 않은 API 키입니다 (X-API-Key 헤더 확인).")
