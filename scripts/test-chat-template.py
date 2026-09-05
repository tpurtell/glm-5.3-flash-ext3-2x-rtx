#!/usr/bin/env python3
"""CPU-only template identity/render gate; never load model weights or images."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

from transformers import AutoProcessor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument(
        "--expected-sha256",
        default="0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5",
    )
    args = parser.parse_args()
    raw = (args.model_dir / "chat_template.jinja").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    assert digest == args.expected_sha256, (digest, args.expected_sha256)
    processor = AutoProcessor.from_pretrained(args.model_dir, local_files_only=True)
    assert processor.chat_template == raw.decode(), "Processor selected another template"
    tools = [{"type": "function", "function": {
        "name": "lookup", "description": "Look up a named item.",
        "parameters": {"type": "object", "properties": {"key": {"type": "string"}}},
    }}]

    def render(messages):
        return processor.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True,
        )

    user = {"role": "user", "content": "Look up A and B."}
    calls = [{"id": ident, "type": "function", "function": {
        "name": "lookup", "arguments": {"key": key},
    }} for ident, key in [("call_a", "A"), ("call_b", "B")]]
    assistant = {"role": "assistant", "content": None, "tool_calls": calls}
    results = [
        {"role": "tool", "tool_call_id": "call_b", "content": "RESULT_B"},
        {"role": "tool", "tool_call_id": "call_a", "content": "RESULT_A"},
    ]
    checks = {}
    plain = render([user])
    checks["ordinary_chat_default_thinking"] = plain.endswith("<think>")
    checks["default_reasoning_effort_max"] = "Reasoning Effort: Max" in plain
    null = render([user, assistant])
    checks["null_assistant_content"] = "None" not in null and null.count("<tool_call>lookup") == 2
    ordered = render([user, assistant, *results])
    checks["out_of_order_tool_results"] = ordered.index("RESULT_A") < ordered.index("RESULT_B")
    for label in ("unknown_id", "missing_id", "duplicate_result_id", "duplicate_call_id"):
        a, r = copy.deepcopy(assistant), copy.deepcopy(results)
        if label == "unknown_id":
            r[0]["tool_call_id"] = "unknown"
        elif label == "missing_id":
            del r[0]["tool_call_id"]
        elif label == "duplicate_result_id":
            r[0]["tool_call_id"] = r[1]["tool_call_id"]
        else:
            a["tool_calls"][1]["id"] = a["tool_calls"][0]["id"]
        fallback = render([user, a, *r])
        checks[label + "_fallback"] = fallback.index("RESULT_B") < fallback.index("RESULT_A")
    listed = render([user, assistant, {"role": "tool", "content": [
        {"tool_call_id": "call_b", "output": "RESULT_B"},
        {"tool_call_id": "call_a", "output": "RESULT_A"},
    ]}])
    checks["list_tool_outputs_reordered"] = listed.index("RESULT_A") < listed.index("RESULT_B")
    reasoning = render([
        user,
        {"role": "assistant", "content": "First answer", "reasoning_content": "REASONING_SENTINEL"},
        {"role": "user", "content": "Continue."},
    ])
    checks["reasoning_preserved_across_follow_up"] = "REASONING_SENTINEL" in reasoning
    images = render([{"role": "user", "content": [
        {"type": "text", "text": "Describe these images."},
        *[{"type": "image", "url": f"https://example.invalid/{i}.png"} for i in range(16)],
    ]}])
    checks["sixteen_image_placeholders"] = images.count("<|begin_of_image|><|image|><|end_of_image|>") == 16
    receipt = {
        "schema": "glm53-chat-template-render-gate.v1",
        "chat_template_sha256": digest,
        "processor": type(processor).__name__,
        "checks": checks,
        "gpu_inference_performed": False,
        "passed": all(checks.values()),
    }
    print(json.dumps(receipt, indent=2))
    assert receipt["passed"], "Template render regression"


if __name__ == "__main__":
    main()
