import asyncio
import uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from config import settings

ALGORITHM = "HS256"
EXPIRE_MINUTES = 60 * 24  # 1 day


def _hash(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def _verify(password: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), hashed.encode())
    except ValueError:
        # e.g. a password longer than bcrypt's 72-byte limit: it can't match
        return False


# bcrypt is CPU-heavy, so run it in a thread to avoid blocking the event loop
async def hash_password(password: str) -> str:
    return await asyncio.to_thread(_hash, password)


async def verify_password(password: str, hashed: str) -> bool:
    return await asyncio.to_thread(_verify, password, hashed)


def create_access_token(user_id: uuid.UUID) -> str:
    now = datetime.now(timezone.utc)
    payload = {"sub": str(user_id), "iat": now, "exp": now + timedelta(minutes=EXPIRE_MINUTES)}
    return jwt.encode(payload, settings.jwt_secret, algorithm=ALGORITHM)


def decode_token(token: str) -> uuid.UUID | None:
    try:
        data = jwt.decode(token, settings.jwt_secret, algorithms=[ALGORITHM])
        return uuid.UUID(data["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        return None