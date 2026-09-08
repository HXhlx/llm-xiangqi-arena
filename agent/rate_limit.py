"""Shared RPM/TPM sliding-window limiter (Redis or in-memory).

Process-local thread pools cannot be shared across API restarts or extra
workers. Counters live under ``xiangqi:rl:`` so checkpoint wipe
(``xiangqi:lg:``) does not reset them.
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Protocol

WINDOW_SEC = 60.0
KEY_PREFIX = "xiangqi:rl:"

DEFAULT_SITE_RPM = 0
DEFAULT_MODEL_RPM = 300
DEFAULT_SITE_TPM = 0
DEFAULT_MODEL_TPM = 0

_LUA_TRY_ACQUIRE = r"""
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local n = tonumber(ARGV[3])
local uniq = ARGV[4]
local cutoff = now - window
local must_wait = 0
local blocked = 0
local used_out = {}

for i = 1, n do
  local key = KEYS[i]
  local limit = tonumber(ARGV[4 + (i * 2) - 1])
  local weight = tonumber(ARGV[4 + (i * 2)])
  redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)
  local members = redis.call('ZRANGE', key, 0, -1)
  local used = 0
  local oldest = nil
  for j = 1, #members do
    local member = members[j]
    local wstr = string.match(member, '^[^:]+:([^:]+):')
    local w = tonumber(wstr) or 1
    used = used + w
    local score = tonumber(redis.call('ZSCORE', key, member))
    if score ~= nil and (oldest == nil or score < oldest) then
      oldest = score
    end
  end
  used_out[i] = used
  if weight > 0 and limit > 0 and (used + weight) > limit then
    blocked = 1
    local wait = 0.05
    if oldest ~= nil then
      wait = oldest + window - now
      if wait < 0 then
        wait = 0.05
      end
    end
    if wait > must_wait then
      must_wait = wait
    end
  end
end

if blocked == 1 then
  local out = {0, tostring(must_wait)}
  for i = 1, n do
    out[#out + 1] = used_out[i]
  end
  return out
end

for i = 1, n do
  local key = KEYS[i]
  local weight = tonumber(ARGV[4 + (i * 2)])
  if weight > 0 then
    local member = string.format('%s:%d:%s:%d', tostring(now), weight, uniq, i)
    redis.call('ZADD', key, now, member)
    redis.call('PEXPIRE', key, math.floor(window * 1000) + 2000)
  end
end

local out = {1, '0'}
for i = 1, n do
  out[#out + 1] = used_out[i]
end
return out
"""

_limiter: Optional["RateLimiter"] = None


def parse_limit(raw: Any, default: int) -> int:
    """Non-negative int. 0 means unlimited. Invalid values fall back to default."""
    if raw is None:
        return max(0, int(default))
    if isinstance(raw, str) and not raw.strip():
        return max(0, int(default))
    try:
        return max(0, int(str(raw).strip()))
    except (TypeError, ValueError):
        return max(0, int(default))


def sanitize_preset(preset: str) -> str:
    text = (preset or "").strip() or "unknown"
    chars = []
    for ch in text:
        if ch.isalnum() or ch in "-_.":
            chars.append(ch)
        else:
            chars.append("_")
    return "".join(chars)[:80] or "unknown"


def rpm_site_key() -> str:
    return f"{KEY_PREFIX}rpm:site"


def rpm_model_key(preset: str) -> str:
    return f"{KEY_PREFIX}rpm:model:{sanitize_preset(preset)}"


def tpm_site_key() -> str:
    return f"{KEY_PREFIX}tpm:site"


def tpm_model_key(preset: str) -> str:
    return f"{KEY_PREFIX}tpm:model:{sanitize_preset(preset)}"


@dataclass(frozen=True)
class RateLimitConfig:
    site_rpm: int = DEFAULT_SITE_RPM
    site_tpm: int = DEFAULT_SITE_TPM
    default_model_rpm: int = DEFAULT_MODEL_RPM
    default_model_tpm: int = DEFAULT_MODEL_TPM
    model_rpm: Mapping[str, int] = field(default_factory=dict)
    model_tpm: Mapping[str, int] = field(default_factory=dict)

    def model_rpm_for(self, preset: str) -> int:
        name = sanitize_preset(preset)
        if name in self.model_rpm:
            return parse_limit(self.model_rpm[name], self.default_model_rpm)
        raw = (preset or "").strip()
        if raw in self.model_rpm:
            return parse_limit(self.model_rpm[raw], self.default_model_rpm)
        return self.default_model_rpm

    def model_tpm_for(self, preset: str) -> int:
        name = sanitize_preset(preset)
        if name in self.model_tpm:
            return parse_limit(self.model_tpm[name], self.default_model_tpm)
        raw = (preset or "").strip()
        if raw in self.model_tpm:
            return parse_limit(self.model_tpm[raw], self.default_model_tpm)
        return self.default_model_tpm


def _optional_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_rate_limit_config(
    env: Mapping[str, str] | None = None,
    yaml_data: Mapping[str, Any] | None = None,
) -> RateLimitConfig:
    src = env if env is not None else os.environ
    data: Mapping[str, Any]
    if yaml_data is None:
        root = Path(__file__).resolve().parents[1]
        data = _optional_yaml(root / "config.yaml")
    else:
        data = yaml_data

    yaml_rl = data.get("rate_limit") if isinstance(data.get("rate_limit"), dict) else {}
    site_rpm = parse_limit(yaml_rl.get("rpm"), DEFAULT_SITE_RPM)
    if str(src.get("XIANGQI_RPM") or "").strip():
        site_rpm = parse_limit(src.get("XIANGQI_RPM"), DEFAULT_SITE_RPM)
    site_tpm = parse_limit(yaml_rl.get("tpm"), DEFAULT_SITE_TPM)
    if str(src.get("XIANGQI_TPM") or "").strip():
        site_tpm = parse_limit(src.get("XIANGQI_TPM"), DEFAULT_SITE_TPM)
    default_model_rpm = parse_limit(yaml_rl.get("model_rpm"), DEFAULT_MODEL_RPM)
    if str(src.get("XIANGQI_MODEL_RPM") or "").strip():
        default_model_rpm = parse_limit(src.get("XIANGQI_MODEL_RPM"), DEFAULT_MODEL_RPM)
    default_model_tpm = parse_limit(yaml_rl.get("model_tpm"), DEFAULT_MODEL_TPM)
    if str(src.get("XIANGQI_MODEL_TPM") or "").strip():
        default_model_tpm = parse_limit(src.get("XIANGQI_MODEL_TPM"), DEFAULT_MODEL_TPM)

    model_rpm: dict[str, int] = {}
    model_tpm: dict[str, int] = {}
    models = data.get("models") or []
    if isinstance(models, list):
        for row in models:
            if not isinstance(row, dict) or not row.get("name"):
                continue
            name = str(row["name"])
            if "rpm" in row:
                model_rpm[name] = parse_limit(row.get("rpm"), default_model_rpm)
            if "tpm" in row:
                model_tpm[name] = parse_limit(row.get("tpm"), default_model_tpm)
    return RateLimitConfig(
        site_rpm=site_rpm,
        site_tpm=site_tpm,
        default_model_rpm=default_model_rpm,
        default_model_tpm=default_model_tpm,
        model_rpm=model_rpm,
        model_tpm=model_tpm,
    )


@dataclass(frozen=True)
class Reservation:
    key: str
    limit: int
    weight: int


@dataclass
class WindowEntry:
    ts: float
    weight: int
    member: str


@dataclass
class AcquireResult:
    ok: bool
    wait_sec: float = 0.0
    used: dict[str, int] = field(default_factory=dict)


def prune_window(
    entries: list[WindowEntry], now: float, window: float
) -> list[WindowEntry]:
    cutoff = now - window
    return [entry for entry in entries if entry.ts > cutoff]


def window_used(entries: list[WindowEntry]) -> int:
    return sum(entry.weight for entry in entries)


def try_reserve(
    entries: list[WindowEntry],
    now: float,
    window: float,
    limit: int,
    weight: int,
) -> tuple[bool, float, int, list[WindowEntry]]:
    live = prune_window(entries, now, window)
    used = window_used(live)
    if weight <= 0:
        return True, 0.0, used, live
    if limit <= 0 or used + weight <= limit:
        return True, 0.0, used, live
    oldest = min(live, key=lambda item: item.ts)
    wait = max(0.05, oldest.ts + window - now)
    return False, wait, used, live


class WindowStore(Protocol):
    async def try_acquire(
        self,
        reservations: list[Reservation],
        now: float,
        window: float,
    ) -> AcquireResult: ...


class MemoryWindowStore:
    """In-process store. Tests and Redis-down fallback share this algorithm."""

    def __init__(self) -> None:
        self.windows: dict[str, list[WindowEntry]] = {}
        self._lock = asyncio.Lock()
        self._seq = 0

    async def try_acquire(
        self,
        reservations: list[Reservation],
        now: float,
        window: float,
    ) -> AcquireResult:
        async with self._lock:
            used: dict[str, int] = {}
            wait = 0.0
            blocked = False
            live_map: dict[str, list[WindowEntry]] = {}
            for item in reservations:
                live = prune_window(list(self.windows.get(item.key) or []), now, window)
                ok, need_wait, count, live = try_reserve(
                    live, now, window, item.limit, item.weight
                )
                used[item.key] = count
                live_map[item.key] = live
                if not ok:
                    blocked = True
                    wait = max(wait, need_wait)
            if blocked:
                for key, live in live_map.items():
                    self.windows[key] = live
                return AcquireResult(ok=False, wait_sec=wait, used=used)
            self._seq += 1
            for item in reservations:
                live = live_map[item.key]
                if item.weight > 0:
                    live.append(
                        WindowEntry(
                            ts=now,
                            weight=item.weight,
                            member=f"{now}:{item.weight}:{self._seq}:{item.key}",
                        )
                    )
                self.windows[item.key] = live
            return AcquireResult(ok=True, wait_sec=0.0, used=used)


class RedisWindowStore:
    def __init__(self, client: Any) -> None:
        self._client = client
        self._script = None

    async def _eval(
        self,
        keys: list[str],
        args: list[Any],
    ) -> list[Any]:
        if self._script is None:
            self._script = self._client.register_script(_LUA_TRY_ACQUIRE)
        return await self._script(keys=keys, args=args)

    async def try_acquire(
        self,
        reservations: list[Reservation],
        now: float,
        window: float,
    ) -> AcquireResult:
        if not reservations:
            return AcquireResult(ok=True)
        keys = [item.key for item in reservations]
        args: list[Any] = [now, window, len(reservations), uuid.uuid4().hex]
        for item in reservations:
            args.extend([item.limit, item.weight])
        raw = await self._eval(keys, args)
        ok = int(raw[0]) == 1
        wait_sec = float(raw[1])
        used = {
            reservations[idx].key: int(raw[2 + idx])
            for idx in range(len(reservations))
            if 2 + idx < len(raw)
        }
        return AcquireResult(ok=ok, wait_sec=wait_sec, used=used)

    async def close(self) -> None:
        close = getattr(self._client, "aclose", None) or getattr(self._client, "close", None)
        if close is None:
            return
        result = close()
        if asyncio.iscoroutine(result):
            await result


class RateLimiter:
    def __init__(
        self,
        store: WindowStore,
        config: RateLimitConfig | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.store = store
        self.config = config or RateLimitConfig()
        self.clock = clock or time.time

    def _rpm_reservations(
        self, preset: str, rpm: int | None
    ) -> tuple[list[Reservation], int, int]:
        model_rpm = (
            self.config.model_rpm_for(preset)
            if rpm is None
            else parse_limit(rpm, self.config.default_model_rpm)
        )
        site_rpm = self.config.site_rpm
        items: list[Reservation] = []
        if site_rpm > 0:
            items.append(Reservation(key=rpm_site_key(), limit=site_rpm, weight=1))
        if model_rpm > 0:
            items.append(Reservation(key=rpm_model_key(preset), limit=model_rpm, weight=1))
        return items, model_rpm, site_rpm

    def _tpm_reservations(
        self, preset: str, tokens: int, tpm: int | None
    ) -> list[Reservation]:
        if tokens <= 0:
            return []
        model_tpm = (
            self.config.model_tpm_for(preset)
            if tpm is None
            else parse_limit(tpm, self.config.default_model_tpm)
        )
        out: list[Reservation] = []
        if self.config.site_tpm > 0:
            out.append(
                Reservation(key=tpm_site_key(), limit=self.config.site_tpm, weight=tokens)
            )
        if model_tpm > 0:
            out.append(
                Reservation(key=tpm_model_key(preset), limit=model_tpm, weight=tokens)
            )
        return out

    def _log_wait(
        self,
        preset: str,
        wait: float,
        used: Mapping[str, int],
        model_rpm: int,
        site_rpm: int,
    ) -> None:
        model_used = used.get(rpm_model_key(preset), 0)
        site_used = used.get(rpm_site_key(), 0)
        print(
            f"  [rate] {preset} wait {wait:.1f}s rpm "
            f"model={model_used}/{model_rpm} site={site_used}/{site_rpm}"
        )

    async def acquire(
        self,
        preset: str,
        *,
        tokens: int = 0,
        rpm: int | None = None,
        tpm: int | None = None,
    ) -> None:
        name = sanitize_preset(preset)
        rpm_items, model_rpm, site_rpm = self._rpm_reservations(name, rpm)
        reservations = list(rpm_items)
        reservations.extend(self._tpm_reservations(name, int(tokens or 0), tpm))
        logged = False
        while True:
            result = await self.store.try_acquire(reservations, self.clock(), WINDOW_SEC)
            if result.ok:
                return
            wait = max(0.05, min(float(result.wait_sec or WINDOW_SEC), WINDOW_SEC))
            if not logged:
                self._log_wait(name, wait, result.used, model_rpm, site_rpm)
                logged = True
            await asyncio.sleep(wait)

    async def record_tokens(
        self,
        preset: str,
        tokens: int,
        *,
        tpm: int | None = None,
    ) -> None:
        name = sanitize_preset(preset)
        reservations = self._tpm_reservations(name, int(tokens or 0), tpm)
        if not reservations:
            return
        await self.store.try_acquire(reservations, self.clock(), WINDOW_SEC)


def get_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter(MemoryWindowStore(), load_rate_limit_config())
    return _limiter


def set_limiter(limiter: RateLimiter | None) -> None:
    global _limiter
    _limiter = limiter


def reset_limiter() -> None:
    set_limiter(None)


async def init_rate_limiter(
    yaml_data: Mapping[str, Any] | None = None,
    env: Mapping[str, str] | None = None,
) -> RateLimiter:
    config = load_rate_limit_config(env=env, yaml_data=yaml_data)
    store: WindowStore
    try:
        import redis.asyncio as redis_async

        from agent.memory import get_redis_url

        client = redis_async.from_url(get_redis_url(), decode_responses=True)
        await client.ping()
        store = RedisWindowStore(client)
    except Exception as exc:
        print(f"  [rate] Redis store unavailable ({exc}); using in-process memory")
        store = MemoryWindowStore()
    limiter = RateLimiter(store, config)
    set_limiter(limiter)
    print(
        f"  [rate] ready site_rpm={config.site_rpm} model_rpm={config.default_model_rpm} "
        f"site_tpm={config.site_tpm} model_tpm={config.default_model_tpm} "
        f"store={type(store).__name__}"
    )
    return limiter


async def close_rate_limiter() -> None:
    limiter = _limiter
    store = getattr(limiter, "store", None) if limiter is not None else None
    closer = getattr(store, "close", None)
    if closer is not None:
        result = closer()
        if asyncio.iscoroutine(result):
            await result
    reset_limiter()


async def acquire_outbound(
    preset: str,
    *,
    tokens: int = 0,
    rpm: int | None = None,
    tpm: int | None = None,
) -> None:
    await get_limiter().acquire(preset, tokens=tokens, rpm=rpm, tpm=tpm)


async def record_outbound_tokens(
    preset: str,
    tokens: int,
    *,
    tpm: int | None = None,
) -> None:
    await get_limiter().record_tokens(preset, tokens, tpm=tpm)
