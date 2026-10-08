"""Opt-in Hermes adapter. Does not intercept built-in reviewer or explicit user saves."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from .core import Memory
from .models import Policy, Task
from .providers import LocalGate, LocalLLMGate, OpenRouterGate, TypeSafeGate
from .store import Store

logger = logging.getLogger(__name__)


def register(ctx):
    def pre_llm_call(**kwargs):
        if ctx.get_config("enabled", False) is not True:
            return None
        text = kwargs.get("user_message")
        # Non-text messages are intentionally not flattened or sent externally.
        if not isinstance(text, str) or not text.strip() or len(text) > 16000:
            return None
        # Require an operator-chosen namespace, scoped again per sender/session.
        namespace = ctx.get_config("namespace", "")
        database = ctx.get_config("database", "")
        owner = kwargs.get("sender_id") or kwargs.get("session_id")
        if not namespace or not database or not owner:
            return None
        try:
            suffix = hashlib.sha256(str(owner).encode()).hexdigest()[:16]
            scope = f"{namespace}/{suffix}"
            provider = ctx.get_config("provider", "local")
            if provider not in {"openrouter", "typesafe", "local", "local-llm"}:
                return None
            options = {"timeout": 5.0}
            if ctx.get_config("decision_model", ""):
                options["model"] = ctx.get_config("decision_model")
            if provider.startswith("local") and ctx.get_config("endpoint", ""):
                options["endpoint"] = ctx.get_config("endpoint")
            gate = {
                "openrouter": OpenRouterGate,
                "typesafe": TypeSafeGate,
                "local": LocalGate,
                "local-llm": LocalLLMGate,
            }[provider](**options)
            task = Task(
                id=hashlib.sha256(str(kwargs.get("turn_id", text)).encode()).hexdigest(),
                namespace=scope,
                text=text,
                tools=ctx.get_config("tools", []),
                environment=ctx.get_config("environment", {}),
            )
            with Store(Path(database).expanduser()) as store:
                result = Memory(
                    store,
                    gate,
                    Policy(shortlist=3, max_lessons=3),
                    shadow=ctx.get_config("shadow", True) is not False,
                ).wake(task)
            return {"context": result["context"]} if result["context"] else None
        except Exception as exc:
            logger.warning("jev-memory skipped: %s", type(exc).__name__)
            return None

    ctx.register_hook("pre_llm_call", pre_llm_call)
