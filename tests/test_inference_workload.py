import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "inference_workload", Path(__file__).resolve().parents[1] / "benchmarks/inference_workload.py"
)
workload = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workload)


class Tokenizer:
    chat_template = "test"
    bos_token_id = 1
    eos_token_id = 2

    def __call__(self, text, **kwargs):
        return {"input_ids": list(text.encode())}

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        return list("".join(message["content"] for message in messages).encode())


@pytest.mark.parametrize("format_name", ["messages", "conversations"])
def test_reference_answer_is_not_in_prompt(format_name):
    tokenizer = Tokenizer()
    row = {"messages": [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]}
    if format_name == "conversations":
        row = {"conversations": [{"from": "human", "value": "question"}, {"from": "gpt", "value": "answer"}]}
    encoded = workload.encode_record(row, tokenizer, 100)
    assert encoded == {"input_ids": list(b"question"), "output_len": 6}
    assert tokenizer.messages == [{"role": "user", "content": "question"}]


def test_empty_prompt_and_output_cap():
    assert workload.encode_record({"prompt": "", "output_len": 20}, Tokenizer(), 8) == {
        "input_ids": [1],
        "output_len": 8,
    }


def test_base_model_uses_single_user_text_without_inventing_a_template():
    tokenizer = Tokenizer()
    tokenizer.chat_template = None
    result = workload.encode_record({"messages": [{"role": "user", "content": "question"}]}, tokenizer, 8)
    assert result["input_ids"] == list(b"question")
    assert not hasattr(tokenizer, "messages")


def test_sharegpt_retains_conversation_context_but_not_the_reference_answer():
    tokenizer = Tokenizer()
    row = {
        "conversations": [
            {"from": "gpt", "value": "previous answer"},
            {"from": "human", "value": "followup"},
            {"from": "gpt", "value": "reference"},
        ]
    }
    result = workload.encode_record(row, tokenizer, 20)
    assert tokenizer.messages == [
        {"role": "assistant", "content": "previous answer"},
        {"role": "user", "content": "followup"},
    ]
    assert result["output_len"] == len("reference")


def test_sampling_is_reproducible_and_warmup_disjoint(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join(json.dumps({"prompt": str(i)}) for i in range(20)))
    args = SimpleNamespace(
        dataset=path, seed=42, num_requests=5, warmup_requests=2, seq_len=8, max_input_tokens=None, request_rate=2.0
    )
    first = workload.prepare_workload(args, Tokenizer())
    assert first == workload.prepare_workload(args, Tokenizer())
    warmup, measured, filtered = first
    assert len(warmup) == 2 and len(measured) == 5
    assert filtered == {"too_long": 0, "invalid": 0, "invalid_examples": []}
    assert not {r["source_index"] for r in warmup} & {r["source_index"] for r in measured}
    assert len({r["source_index"] for r in measured}) == 5
    assert measured[0]["arrival_offset_s"] == 0
    assert all(a["arrival_offset_s"] < b["arrival_offset_s"] for a, b in zip(measured, measured[1:]))
    args.num_requests = 19
    with pytest.raises(ValueError, match="No oversampling"):
        workload.prepare_workload(args, Tokenizer())


def test_malformed_records_are_counted_not_oversampled(tmp_path):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps([{"conversations": []}]))
    args = SimpleNamespace(
        dataset=path, seed=42, num_requests=1, warmup_requests=0, seq_len=8, max_input_tokens=None, request_rate=None
    )
    with pytest.raises(ValueError, match="'invalid': 1"):
        workload.prepare_workload(args, Tokenizer())


def test_summary_uses_wall_time_and_counts_failures():
    rows = [
        {"success": True, "arrival_s": 0, "started_s": 0.5, "token_times_s": [2, 3], "output_tokens": 2},
        {"success": True, "arrival_s": 1, "started_s": 1.25, "token_times_s": [2.5], "output_tokens": 1},
        {"success": False, "output_tokens": 4},
    ]
    result = workload.summarize(rows, duration=4, offered=3)
    assert result["output_tokens_per_s"] == 0.75
    assert result["requests_per_s"] == 0.5
    assert result["failure_rate"] == pytest.approx(1 / 3)
    assert result["failed_requests"] == 1
    assert result["metrics"]["ttft_ms"]["mean"] == 1750
    assert result["metrics"]["queue_ms"]["mean"] == 375
    assert result["metrics"]["tpot_ms"]["count"] == 1
    assert result["metrics"]["itl_ms"]["mean"] == 1000
    json.dumps(workload.summarize([], 0, 3), allow_nan=False)


@pytest.mark.parametrize("ignore_eos, count", [(False, 2), (True, 3)])
def test_request_stops_at_eos_and_uses_a_fresh_session(monkeypatch, ignore_eos, count):
    class Model:
        generation_config = SimpleNamespace(eos_token_id=[7, 9])
        sessions = 0

        @contextmanager
        def inference_session(self, **kwargs):
            self.sessions += 1
            self.inputs = []
            yield self

        def generate(self, ids, **kwargs):
            self.inputs.append(ids)
            return torch.tensor([[5, 7, 6][len(self.inputs) - 1]]).reshape(1, 1)

    model = Model()
    monkeypatch.setattr(workload, "_MODEL", model)
    monkeypatch.setattr(workload, "_OPTIONS", SimpleNamespace(client_device="cpu", ignore_eos=ignore_eos))
    request = {"request_id": 1, "source_index": 0, "input_ids": [1, 2], "output_len": 3}
    for _ in range(2):
        result = workload.run_request(request, workload.perf_counter())
        assert result["success"] and result["output_tokens"] == count
        assert result["stop_reason"] == ("length" if ignore_eos else "eos")
        assert model.inputs[0].tolist() == [[1, 2]]
        assert all(ids is None for ids in model.inputs[1:])
    assert model.sessions == 2
