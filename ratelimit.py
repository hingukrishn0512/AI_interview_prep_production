"""Per-user rate limiting backed by Redis (fixed windows).

Each user gets a counter per window, e.g. rl:chat_min:<user_id>:<window>.
INCR + EXPIRE run in one transaction, so a counter can never be left without
an expiry. If Redis is down, requests are allowed (fail open) and a warning is
logged, so a Redis outage doesn't take the whole app down.
"""
import logging
import time

import redis.asyncio as redis
import structlog
from fastapi import Depends, HTTPException, Request

from auth import get_current_user
from config import settings
from db.models import User

logger = logging.getLogger("interview_coach.ratelimit")

redis_client = redis.from_url(settings.redis_url, decode_responses=True)


def rate_limit(name: str, limit: int, window_seconds: int, message: str):
    """Builds a FastAPI dependency that returns the logged-in user, or raises 429."""

    async def dependency(
        request: Request,
        user: User = Depends(get_current_user),
    ) -> User:
        # Attach the user to this request's logs.
        request.state.user_id = str(user.id)
        structlog.contextvars.bind_contextvars(user_id=str(user.id))

        now = int(time.time())
        key = f"rl:{name}:{user.id}:{now // window_seconds}"
        try:
            async with redis_client.pipeline(transaction=True) as pipe:
                pipe.incr(key)
                pipe.expire(key, window_seconds)
                count, _ = await pipe.execute()
        except Exception as e:
            logger.warning("Rate limiter unavailable, allowing request: %s", e)
            return user

        if count > limit:
            retry_after = max(window_seconds - (now % window_seconds), 1)
            logger.info(
                "rate_limited limit=%s window=%s name=%s retry_after=%s",
                limit, window_seconds, name, retry_after,
            )
            raise HTTPException(
                status_code=429,
                detail=f"{message} Try again in {retry_after} seconds.",
                headers={"Retry-After": str(retry_after)},
            )
        return user

    return dependency


# Tune these as real usage shows up.
chat_per_minute = rate_limit("chat_min", 10, 60, "You're sending messages too fast.")
chat_per_day = rate_limit("chat_day", 150, 24 * 60 * 60, "You've reached today's message limit.")
session_per_hour = rate_limit("session_hour", 10, 60 * 60, "You've started too many sessions.")