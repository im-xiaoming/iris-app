from __future__ import annotations

import json
from typing import Any

import redis


class RedisCache:
    def __init__(self, host: str, port: int, db: int, ttl_seconds: int, prefix: str) -> None:
        self.client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
        self.ttl_seconds = ttl_seconds
        self.prefix = prefix

    def _build_key(self, fingerprint: str) -> str:
        return f"{self.prefix}:{fingerprint}"

    def get(self, fingerprint: str) -> dict[str, Any] | None:
        value = self.client.get(self._build_key(fingerprint))
        if value is None:
            return None
        return json.loads(value)

    def set(self, fingerprint: str, payload: dict[str, Any]) -> None:
        self.client.setex(
            self._build_key(fingerprint),
            self.ttl_seconds,
            json.dumps(payload),
        )
