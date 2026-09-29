"""Phase 0: confirm API access before building the real harness."""
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY")


def check_jev():
    if not OPENROUTER_KEY:
        print("[jev]      SKIP - no OPENROUTER_API_KEY")
        return
    resp = httpx.post(
        "https://openrouter.ai/api/alpha/decisions",
        headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
        json={
            "model": "typesafe/jev-1.13",
            "state": "The customer says the app crashes every time they open settings.",
            "questions": {
                "is_bug": {
                    "type": "noul",
                    "instructions": "Is the customer reporting a software defect?",
                    "criteria": {
                        "true": "Describes broken or unexpected product behavior.",
                        "false": "Asking a question or requesting a feature.",
                    },
                }
            },
        },
        timeout=30,
    )
    print(f"[jev]      status={resp.status_code}")
    print(f"           body={resp.text[:500]}")


def check_openrouter_chat_model(model_id, label):
    if not OPENROUTER_KEY:
        print(f"[{label}] SKIP - no OPENROUTER_API_KEY")
        return
    resp = httpx.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
        json={
            "model": model_id,
            "messages": [{"role": "user", "content": "Reply with exactly: ok"}],
            "max_tokens": 10,
        },
        timeout=30,
    )
    print(f"[{label}] status={resp.status_code}")
    print(f"           body={resp.text[:500]}")


if __name__ == "__main__":
    check_jev()
    check_openrouter_chat_model("openai/gpt-6-luna", "gpt-6-luna")
    check_openrouter_chat_model("anthropic/claude-haiku-4.5", "haiku-4.5")
    check_openrouter_chat_model("anthropic/claude-sonnet-5", "sonnet-5")
