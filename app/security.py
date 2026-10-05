import asyncio
from datetime import UTC, datetime, timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, VerifyMismatchError
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.models import KnowledgeBase, User
from app.settings import Settings

hasher = PasswordHasher()
bearer = HTTPBearer(auto_error=False)


async def hash_password(password: str) -> str:
    return await asyncio.to_thread(hasher.hash, password)


async def verify_password(hashed: str, password: str) -> bool:
    try:
        return await asyncio.to_thread(hasher.verify, hashed, password)
    except (VerifyMismatchError, VerificationError):
        return False


def issue_token(user_id: str, settings: Settings) -> str:
    now = datetime.now(UTC)
    return jwt.encode(
        {
            "sub": user_id,
            "iat": now,
            "exp": now + timedelta(minutes=settings.token_ttl_minutes),
            "iss": settings.jwt_issuer,
            "aud": "rag-api",
        },
        settings.jwt_secret.get_secret_value(),
        algorithm="HS256",
    )


async def current_user(
    request: Request, credentials: HTTPAuthorizationCredentials | None = Depends(bearer)
) -> User:
    if credentials is None:
        raise HTTPException(401, "请先登录")
    settings: Settings = request.app.state.settings
    try:
        payload = jwt.decode(
            credentials.credentials,
            settings.jwt_secret.get_secret_value(),
            algorithms=["HS256"],
            issuer=settings.jwt_issuer,
            audience="rag-api",
            options={"require": ["sub", "exp", "iat"]},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(401, "登录凭证无效或已过期") from None
    user = await User.get_or_none(id=payload["sub"], is_active=True)
    if user is None:
        raise HTTPException(401, "用户不存在")
    return user


async def owned_kb(kb_id: str, user_id: str) -> KnowledgeBase:
    kb = await KnowledgeBase.get_or_none(id=kb_id, owner_id=user_id)
    if kb is None:
        raise HTTPException(404, "知识库不存在")
    return kb
