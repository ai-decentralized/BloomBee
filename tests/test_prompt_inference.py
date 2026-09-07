"""End-to-end distributed generation against a live swarm, with a caller-supplied prompt.

Unlike the parity tests in test_full_model.py (which compare against a local HF reference
model and therefore need the reference weights on the test machine), this test only checks
that the swarm produces a finite, decodable continuation for an arbitrary prompt. It's meant
as a quick manual/CI smoke check that "inference actually works end to end right now" --
tokenizer -> chat template -> distributed generate() -> decode -- against whatever swarm
INITIAL_PEERS/MODEL_NAME point at.

Requires a running swarm (see README.md "Start Worker Servers"). Skipped automatically
(via test_utils) if INITIAL_PEERS/MODEL_NAME aren't set.

Env vars:
    INITIAL_PEERS          (required) space-separated bootstrap multiaddrs, e.g.
                            "/ip4/10.140.81.87/tcp/31340/p2p/Qm..."
    MODEL_NAME              (required) e.g. "unsloth/DeepSeek-V3-bf16"
    PROMPT                  (optional) defaults to "What is the capital of France?"
    MAX_NEW_TOKENS           (optional) defaults to 8 -- kept small so the run fits inside
                            pytest.ini's 600s timeout on a CPU swarm (~15-20s/token observed
                            on a 6-block-per-worker DeepSeek-V3 INT8 CPU swarm). Raise it (and
                            pytest's --timeout) for a longer, more convincing continuation.
    USE_SERVER_TO_SERVER    (optional) "1" to let workers relay directly to each other.
                            Defaults to "0" (client mediates every hop): worker-to-worker
                            relaying was found to hang on a broken leg of the relay chain
                            between two specific workers, invisible to client-side connectivity
                            checks. See benchmarks/benchmark_inference.py's
                            --use_server_to_server flag for the same escape hatch.

Example:
    INITIAL_PEERS="/ip4/10.140.81.87/tcp/31340/p2p/QmNmATs5cVnw36iJ9qfATNXX5wKvsGdRYfrYGYPoLK55J7" \\
    MODEL_NAME="unsloth/DeepSeek-V3-bf16" \\
    PROMPT="Explain gravity in one sentence." \\
    pytest tests/test_prompt_inference.py -s -v
"""

import os
import time

import pytest
import torch
from transformers import AutoTokenizer

from bloombee import AutoDistributedModelForCausalLM
from bloombee.utils.hivemind_compat import get_logger
from test_utils import *

logger = get_logger(__name__)

PROMPT = os.environ.get("PROMPT", "What is the capital of France?")
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "8"))
USE_SERVER_TO_SERVER = os.environ.get("USE_SERVER_TO_SERVER", "0") == "1"


@pytest.mark.forked
def test_prompt_inference():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    messages = [{"role": "user", "content": PROMPT}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    )

    model = AutoDistributedModelForCausalLM.from_pretrained(
        MODEL_NAME,
        initial_peers=INITIAL_PEERS,
        torch_dtype=torch.bfloat16,
        use_server_to_server=USE_SERVER_TO_SERVER,
    )

    start = time.perf_counter()
    outputs = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    elapsed = time.perf_counter() - start

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    logger.info(f"[test_prompt_inference] prompt={PROMPT!r} elapsed={elapsed:.2f}s response={response!r}")

    assert torch.all(torch.isfinite(outputs.float())), "Generation produced non-finite token ids"
    assert generated_ids.shape[0] == MAX_NEW_TOKENS, "Generation stopped short of max_new_tokens unexpectedly"
    assert response.strip() != "", "Decoded response was empty"
