import json
import logging
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import litellm
from pydantic import BaseModel
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from minisweagent.models import GLOBAL_MODEL_STATS
from minisweagent.models.utils.cache_control import set_cache_control

logger = logging.getLogger("litellm_model")


class LitellmModelConfig(BaseModel):
    model_name: str
    model_kwargs: dict[str, Any] = {}
    litellm_model_registry: Path | str | None = os.getenv("LITELLM_MODEL_REGISTRY_PATH")
    set_cache_control: Literal["default_end"] | None = None
    """Set explicit cache control markers, for example for Anthropic models"""
    cost_tracking: Literal["default", "ignore_errors"] = os.getenv("MSWEA_COST_TRACKING", "default")
    """Cost tracking mode for this model. Can be "default" or "ignore_errors" (ignore errors/missing cost info)"""


# Keys consumed locally; never forward to the OpenAI-compatible API.
_LOCAL_MODEL_KWARG_KEYS = frozenset(
    {
        "auto_max_tokens",
        "max_model_len",
        "max_tokens_reserve",
        "max_tokens_cap",
    }
)


def _truthy(val: Any) -> bool:
    if isinstance(val, bool):
        return val
    if val is None:
        return False
    return str(val).strip().lower() in {"1", "true", "yes", "y", "on"}


def _estimate_prompt_tokens(model_name: str, messages: list[dict[str, str]]) -> int:
    try:
        n = litellm.token_counter(model=model_name, messages=messages)
        if isinstance(n, int) and n > 0:
            return n
    except Exception:
        pass
    # Fallback: rough char→token heuristic
    chars = 0
    for msg in messages:
        content = msg.get("content") or ""
        chars += len(content) if isinstance(content, str) else len(str(content))
    return max(1, chars // 4)


class LitellmModel:
    def __init__(self, *, config_class: Callable = LitellmModelConfig, **kwargs):
        self.config = config_class(**kwargs)
        self.cost = 0.0
        self.n_calls = 0
        if self.config.litellm_model_registry and Path(self.config.litellm_model_registry).is_file():
            litellm.utils.register_model(json.loads(Path(self.config.litellm_model_registry).read_text()))

    def _prepare_completion_kwargs(self, messages: list[dict[str, str]], **kwargs) -> dict[str, Any]:
        """Merge config/call kwargs; optionally set max_tokens from remaining context window."""
        merged: dict[str, Any] = dict(self.config.model_kwargs) | dict(kwargs)

        auto = merged.pop("auto_max_tokens", None)
        max_model_len_cfg = merged.pop("max_model_len", None)
        reserve_cfg = merged.pop("max_tokens_reserve", None)
        cap_cfg = merged.pop("max_tokens_cap", None)
        for k in list(merged.keys()):
            if k in _LOCAL_MODEL_KWARG_KEYS:
                merged.pop(k, None)

        # Env fallbacks (useful with hosted_vllm / start_vllm.sh)
        if auto is None:
            auto = os.getenv("MSWEA_AUTO_MAX_TOKENS", "0")
        if max_model_len_cfg is None:
            max_model_len_cfg = os.getenv("MSWEA_MAX_MODEL_LEN") or os.getenv("VLLM_MAX_MODEL_LEN")
        if reserve_cfg is None:
            reserve_cfg = os.getenv("MSWEA_MAX_TOKENS_RESERVE", "0")
        if cap_cfg is None:
            cap_cfg = os.getenv("MSWEA_MAX_TOKENS_CAP")  # optional soft per-step cap

        # Explicit max_tokens from caller/config wins unless auto is on and value is "auto"
        existing = merged.get("max_tokens", None)
        want_auto = _truthy(auto) or (
            isinstance(existing, str) and existing.strip().lower() == "auto"
        )
        if existing is not None and not want_auto and not (
            isinstance(existing, str) and existing.strip().lower() == "auto"
        ):
            return merged

        if not want_auto:
            return merged

        try:
            max_model_len = int(max_model_len_cfg) if max_model_len_cfg is not None else 131072
        except (TypeError, ValueError):
            max_model_len = 131072
        try:
            reserve = max(0, int(reserve_cfg))
        except (TypeError, ValueError):
            reserve = 0

        prompt_tokens = _estimate_prompt_tokens(self.config.model_name, messages)
        remaining = max_model_len - prompt_tokens - reserve
        if remaining < 1:
            raise litellm.exceptions.ContextWindowExceededError(
                message=(
                    f"Prompt too long for auto max_tokens: prompt≈{prompt_tokens}, "
                    f"max_model_len={max_model_len}, reserve={reserve}"
                ),
                model=self.config.model_name,
                llm_provider="hosted_vllm",
            )

        max_tokens = remaining
        if cap_cfg is not None and str(cap_cfg).strip() != "":
            try:
                cap = int(cap_cfg)
                if cap > 0:
                    max_tokens = min(max_tokens, cap)
            except (TypeError, ValueError):
                pass

        merged["max_tokens"] = int(max_tokens)
        logger.info(
            "auto_max_tokens: max_model_len=%s prompt≈%s reserve=%s -> max_tokens=%s%s",
            max_model_len,
            prompt_tokens,
            reserve,
            merged["max_tokens"],
            f" (cap={cap_cfg})" if cap_cfg not in (None, "") else "",
        )
        return merged

    @retry(
        reraise=True,
        stop=stop_after_attempt(int(os.getenv("MSWEA_MODEL_RETRY_STOP_AFTER_ATTEMPT", "10"))),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        retry=retry_if_not_exception_type(
            (
                litellm.exceptions.UnsupportedParamsError,
                litellm.exceptions.NotFoundError,
                litellm.exceptions.PermissionDeniedError,
                litellm.exceptions.ContextWindowExceededError,
                litellm.exceptions.APIError,
                litellm.exceptions.AuthenticationError,
                KeyboardInterrupt,
            )
        ),
    )
    def _query(self, messages: list[dict[str, str]], **kwargs):
        try:
            call_kwargs = self._prepare_completion_kwargs(messages, **kwargs)
            return litellm.completion(
                model=self.config.model_name, messages=messages, **call_kwargs
            )
        except litellm.exceptions.AuthenticationError as e:
            e.message += " You can permanently set your API key with `mini-extra config set KEY VALUE`."
            raise e

    def query(self, messages: list[dict[str, str]], **kwargs) -> dict:
        if self.config.set_cache_control:
            messages = set_cache_control(messages, mode=self.config.set_cache_control)
        response = self._query([{"role": msg["role"], "content": msg["content"]} for msg in messages], **kwargs)
        try:
            cost = litellm.cost_calculator.completion_cost(response, model=self.config.model_name)
            if cost <= 0.0:
                raise ValueError(f"Cost must be > 0.0, got {cost}")
        except Exception as e:
            cost = 0.0
            if self.config.cost_tracking != "ignore_errors":
                msg = (
                    f"Error calculating cost for model {self.config.model_name}: {e}, perhaps it's not registered? "
                    "You can ignore this issue from your config file with cost_tracking: 'ignore_errors' or "
                    "globally with export MSWEA_COST_TRACKING='ignore_errors'. "
                    "Alternatively check the 'Cost tracking' section in the documentation at "
                    "https://klieret.short.gy/mini-local-models. "
                    " Still stuck? Please open a github issue at https://github.com/SWE-agent/mini-swe-agent/issues/new/choose!"
                )
                logger.critical(msg)
                raise RuntimeError(msg) from e
        self.n_calls += 1
        self.cost += cost
        GLOBAL_MODEL_STATS.add(cost)
        return {
            "content": response.choices[0].message.content or "",  # type: ignore
            "extra": {
                "response": response.model_dump(),
            },
        }

    def get_template_vars(self) -> dict[str, Any]:
        return self.config.model_dump() | {"n_model_calls": self.n_calls, "model_cost": self.cost}
