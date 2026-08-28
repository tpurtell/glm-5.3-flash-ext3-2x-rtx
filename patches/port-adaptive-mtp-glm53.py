#!/usr/bin/env python3
"""Wire the feedback-adaptive MTP controller into the pinned vLLM scheduler."""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    print(f"[ok]   {label}")
    return text.replace(old, new, 1)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: port-adaptive-mtp-glm53.py VLLM_ROOT")
    root = Path(sys.argv[1])
    scheduler = root / "v1/core/sched/scheduler.py"
    text = scheduler.read_text()

    text = replace_once(
        text,
        "from vllm.v1.spec_decode.dynamic.utils import "
        "build_dynamic_sd_schedule_lookup\n",
        "from vllm.v1.spec_decode.dynamic.adaptive_mtp import (\n"
        "    AdaptiveMTPController,\n"
        "    adaptive_mtp_enabled,\n"
        ")\n"
        "from vllm.v1.spec_decode.dynamic.utils import "
        "build_dynamic_sd_schedule_lookup\n",
        "import adaptive MTP controller",
    )
    text = replace_once(
        text,
        "            self.use_eagle = speculative_config.use_eagle()\n\n"
        "        # Create the KV cache manager.\n",
        "            self.use_eagle = speculative_config.use_eagle()\n\n"
        "        self.adaptive_mtp_controller: AdaptiveMTPController | None = None\n"
        "        if adaptive_mtp_enabled():\n"
        "            if (\n"
        "                speculative_config is None\n"
        "                or getattr(speculative_config, \"method\", None) != \"mtp\"\n"
        "            ):\n"
        "                raise ValueError(\n"
        "                    \"VLLM_ADAPTIVE_MTP requires MTP speculative decoding\"\n"
        "                )\n"
        "            if self.dynamic_sd_lookup is None:\n"
        "                raise ValueError(\n"
        "                    \"VLLM_ADAPTIVE_MTP requires \"\n"
        "                    \"num_speculative_tokens_per_batch_size so every \"\n"
        "                    \"reachable variable-K CUDA graph is prepared\"\n"
        "                )\n"
        "            self.adaptive_mtp_controller = AdaptiveMTPController.from_env(\n"
        "                self.num_spec_tokens\n"
        "            )\n"
            "            logger.info(\n"
            "                \"Enabled request-local feedback-adaptive MTP \"\n"
            "                \"(K=0..%d, arithmetic-mean batch execution)\",\n"
        "                self.num_spec_tokens,\n"
        "            )\n\n"
        "        # Create the KV cache manager.\n",
        "initialize adaptive MTP controller",
    )
    text = replace_once(
        text,
        "        self.current_step += 1\n"
        "        # NOTE(woosuk) on the scheduling algorithm:\n",
        "        self.current_step += 1\n"
        "        if (\n"
        "            self.adaptive_mtp_controller is not None\n"
        "            and self.finished_req_ids\n"
        "        ):\n"
        "            self.adaptive_mtp_controller.finish(self.finished_req_ids)\n"
        "        # NOTE(woosuk) on the scheduling algorithm:\n",
        "release finished request-local MTP state",
    )
    text = replace_once(
        text,
        "        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:\n"
        "            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[\n"
        "                len(num_scheduled_tokens)\n"
        "            ]\n",
        "        if self.dynamic_sd_lookup is not None and len(num_scheduled_tokens) > 0:\n"
        "            num_spec_tokens_to_schedule = self.dynamic_sd_lookup[\n"
        "                len(num_scheduled_tokens)\n"
        "            ]\n"
        "            if self.adaptive_mtp_controller is not None:\n"
        "                adaptive_request_ids = [\n"
        "                    req_id\n"
        "                    for req_id in num_scheduled_tokens\n"
        "                    if (request := self.requests.get(req_id)) is not None\n"
        "                    and not request.is_prefill_chunk\n"
        "                ]\n"
        "                num_spec_tokens_to_schedule = (\n"
        "                    self.adaptive_mtp_controller.select(\n"
        "                        adaptive_request_ids,\n"
        "                        len(adaptive_request_ids),\n"
        "                        num_spec_tokens_to_schedule,\n"
        "                    )\n"
        "                )\n",
        "select feedback-adaptive runtime K",
    )
    text = replace_once(
        text,
        "        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)\n"
        "        spec_decoding_stats: SpecDecodingStats | None = None\n",
        "        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)\n"
        "        spec_decoding_stats: SpecDecodingStats | None = None\n"
        "        adaptive_mtp_observations: list[tuple[str, int, int]] = []\n",
        "collect per-step adaptive MTP observations",
    )
    text = replace_once(
        text,
        "                num_rejected = num_draft_tokens - num_accepted\n"
        "                # Rejections roll back num_computed_tokens (and, under async\n",
        "                num_rejected = num_draft_tokens - num_accepted\n"
        "                if self.adaptive_mtp_controller is not None:\n"
        "                    num_invalid = (\n"
        "                        scheduler_output.num_invalid_spec_tokens or {}\n"
        "                    ).get(req_id, 0)\n"
        "                    adaptive_mtp_observations.append(\n"
        "                        (\n"
        "                            req_id,\n"
        "                            num_draft_tokens - num_invalid,\n"
        "                            num_accepted,\n"
        "                        )\n"
        "                    )\n"
        "                # Rejections roll back num_computed_tokens (and, under async\n",
        "observe accepted draft prefix",
    )
    text = replace_once(
        text,
        "        if (\n"
        "            stats := self.make_stats(\n",
        "        if (\n"
        "            self.adaptive_mtp_controller is not None\n"
        "            and adaptive_mtp_observations\n"
        "        ):\n"
        "            adaptive_batch_size = len(num_scheduled_tokens)\n"
        "            for req_id, proposed, accepted in adaptive_mtp_observations:\n"
        "                self.adaptive_mtp_controller.observe(\n"
        "                    req_id, adaptive_batch_size, proposed, accepted\n"
        "                )\n\n"
        "        if (\n"
        "            stats := self.make_stats(\n",
        "update adaptive MTP hysteresis",
    )

    scheduler.write_text(text)
    print("feedback-adaptive MTP scheduler port applied")


if __name__ == "__main__":
    main()
