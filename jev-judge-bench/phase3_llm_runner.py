"""Phase 3: same dev-set zero-shot pass, but for the LLM baselines
(Haiku 4.5, GPT-6 Luna) via OpenRouter chat completions - reuses the exact
same state/instructions/criteria already baked into each sample's payload
so the framing is apples-to-apples with the Jev run."""
import json
import os
import re
import time

import httpx
from dotenv import load_dotenv

load_dotenv()
OPENROUTER_KEY = os.environ["OPENROUTER_API_KEY"]
CHAT_URL = "https://openrouter.ai/api/v1/chat/completions"

VERDICT_RE = re.compile(r"VERDICT:\s*(true|false)", re.IGNORECASE)
CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*(\d+)")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def build_judge_prompt(payload, question_key):
    state = payload["state"]
    q = payload["questions"][question_key]
    state_text = "\n".join(f"{k.capitalize()}: {v}" for k, v in state.items())
    return f"""{state_text}

Question: {q['instructions']}

Answer "true" if: {q['criteria']['true']}
Answer "false" if: {q['criteria']['false']}

Respond in exactly this format, nothing else:
VERDICT: true or false
CONFIDENCE: a number from 0 to 100 (your confidence that VERDICT is correct)
REASON: one sentence explaining why"""


def call_llm_judge(model_id, prompt, timeout=60, max_retries=10, max_tokens=200):
    t0 = time.monotonic()
    for attempt in range(max_retries):
        resp = httpx.post(
            CHAT_URL,
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0,
            },
            timeout=timeout,
        )
        if resp.status_code in (403, 429, 500, 502, 503, 529) and attempt < max_retries - 1:
            retry_after = resp.headers.get("retry-after")
            wait = float(retry_after) if retry_after else min(2**attempt, 60)
            time.sleep(wait)
            continue
        break
    latency_ms = (time.monotonic() - t0) * 1000
    if resp.status_code >= 400:
        print(f"  FINAL FAILURE {resp.status_code}: {resp.text[:300]}")
    resp.raise_for_status()
    body = resp.json()

    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message") or {}
    content = message.get("content") or ""
    finish_reason = choice.get("finish_reason")
    content_filtered = finish_reason == "content_filter"
    usage = body.get("usage") or {}

    verdict_match = VERDICT_RE.search(content)
    confidence_match = CONFIDENCE_RE.search(content)
    parse_error = verdict_match is None

    if content_filtered:
        print(f"  content_filter hit, no verdict available: {json.dumps(body)[:300]}")

    return {
        "raw_prompt": prompt,
        "raw_response_text": content,
        "verdict_true": (verdict_match.group(1).lower() == "true") if verdict_match else None,
        "confidence": int(confidence_match.group(1)) if confidence_match else None,
        "parse_error": parse_error,
        "content_filtered": content_filtered,
        "latency_ms": latency_ms,
        "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
        "output_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
        "cost": usage.get("cost", 0.0),
    }


def run_llm_on_file(model_id, in_path, out_path, question_key, ground_truth_key, limit=None, max_tokens=200, resume=True):
    rows = load_jsonl(in_path)
    if limit:
        rows = rows[:limit]

    done_ids = set()
    if resume:
        try:
            done_ids = {r["id"] for r in load_jsonl(out_path)}
        except FileNotFoundError:
            pass
    rows = [r for r in rows if r["id"] not in done_ids]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results = []
    parse_errors = 0
    with open(out_path, "a") as f:
        for i, row in enumerate(rows):
            prompt = build_judge_prompt(row["payload"], question_key)
            call = call_llm_judge(model_id, prompt, max_tokens=max_tokens)
            if call["parse_error"]:
                parse_errors += 1
            result = {
                "id": row["id"],
                "ground_truth": row[ground_truth_key],
                "predicted": call["verdict_true"],
                "confidence": call["confidence"],
                "parse_error": call["parse_error"],
                "content_filtered": call["content_filtered"],
                "latency_ms": call["latency_ms"],
                "input_tokens": call["input_tokens"],
                "output_tokens": call["output_tokens"],
                "cost": call["cost"],
                "raw_prompt": call["raw_prompt"],
                "raw_response_text": call["raw_response_text"],
            }
            results.append(result)
            f.write(json.dumps(result) + "\n")
            f.flush()
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(rows)} done (this run; {len(done_ids)} resumed from prior)")
            time.sleep(0.2)
    total_cost = sum(r["cost"] for r in results)
    print(
        f"[{model_id} | {in_path}] n_this_run={len(results)} resumed={len(done_ids)} "
        f"parse_errors={parse_errors} total_cost_this_run=${total_cost:.5f}"
    )
    return results


if __name__ == "__main__":
    import sys

    which = sys.argv[1] if len(sys.argv) > 1 else "smoke"

    if which == "smoke":
        risk_rows = load_jsonl("data/samples/risk_dev.jsonl")
        prompt = build_judge_prompt(risk_rows[0]["payload"], "is_safe")
        print("=" * 100)
        print("EXACT PROMPT SENT TO LLM JUDGE (risk example)")
        print("=" * 100)
        print(prompt)
        result = call_llm_judge("anthropic/claude-haiku-4.5", prompt)
        print("\nRESPONSE TEXT:")
        print(result["raw_response_text"])
        print(f"\nparsed verdict_true={result['verdict_true']} confidence={result['confidence']}")

    elif which == "run_dev":
        for model_id, tag in [
            ("anthropic/claude-haiku-4.5", "haiku45"),
            ("openai/gpt-6-luna", "luna"),
        ]:
            print(f"\n=== {tag}: risk dev, zero-shot ===")
            run_llm_on_file(
                model_id,
                "data/samples/risk_dev.jsonl",
                f"results/raw/{tag}_zeroshot_risk_dev.jsonl",
                question_key="is_safe",
                ground_truth_key="ground_truth_is_safe",
                max_tokens=600,
            )
            print(f"\n=== {tag}: quality dev, zero-shot ===")
            run_llm_on_file(
                model_id,
                "data/samples/quality_dev.jsonl",
                f"results/raw/{tag}_zeroshot_quality_dev.jsonl",
                question_key="is_high_quality",
                ground_truth_key="ground_truth_is_high_quality",
                max_tokens=600,
            )
    elif which == "run_test":
        model_tag = sys.argv[2]  # "haiku45" or "luna"
        judge = sys.argv[3]  # "risk" or "quality"
        model_id = {
            "haiku45": "anthropic/claude-haiku-4.5",
            "luna": "openai/gpt-6-luna",
        }[model_tag]
        if judge == "risk":
            run_llm_on_file(
                model_id,
                "data/samples/risk_test.jsonl",
                f"results/raw/{model_tag}_zeroshot_risk_test.jsonl",
                question_key="is_safe",
                ground_truth_key="ground_truth_is_safe",
                max_tokens=600,
            )
        else:
            run_llm_on_file(
                model_id,
                "data/samples/quality_test.jsonl",
                f"results/raw/{model_tag}_zeroshot_quality_test.jsonl",
                question_key="is_high_quality",
                ground_truth_key="ground_truth_is_high_quality",
                max_tokens=600,
            )

    elif which == "run_dev_luna_only":
        run_llm_on_file(
            "openai/gpt-6-luna",
            "data/samples/risk_dev.jsonl",
            "results/raw/luna_zeroshot_risk_dev.jsonl",
            question_key="is_safe",
            ground_truth_key="ground_truth_is_safe",
            max_tokens=600,
        )
        run_llm_on_file(
            "openai/gpt-6-luna",
            "data/samples/quality_dev.jsonl",
            "results/raw/luna_zeroshot_quality_dev.jsonl",
            question_key="is_high_quality",
            ground_truth_key="ground_truth_is_high_quality",
            max_tokens=600,
        )
