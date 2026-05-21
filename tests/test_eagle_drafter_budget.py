import torch
from transformers import GenerationConfig
from transformers.generation import LogitsProcessorList, StoppingCriteriaList

from bloombee.models.llama.eagle_drafter import (
    EAGLEDrafter,
    _default_draft_token_budget,
    _default_eagle_topk_per_step,
    _eagle2_max_candidate_depth,
)
from bloombee.models.llama.spe_dec_tree import SpeculativeTree, linearize_tree_with_positions
from bloombee.models.llama.speculative_model import (
    DistributedLlamaForSpeculativeGeneration,
    _attention_mask_from_seq_lengths,
    _cap_valid_lengths_to_remaining,
    _eos_token_ids,
    _merge_generation_config_kwargs,
    _project_lm_head,
    _speculative_session_max_length,
)


def test_eagle_legacy_budget_matches_chain_draft_depth():
    assert _default_draft_token_budget(beam_width=1, max_depth=5) == 5


def test_eagle_legacy_budget_matches_scalar_tree_plan():
    assert _default_draft_token_budget(beam_width=3, max_depth=4) == 12


def test_eagle_legacy_budget_matches_sequoia_width_plan():
    assert _default_draft_token_budget(beam_width=[2, 3, 1], max_depth=3) == 14


def test_eagle_paper_budget_uses_official_depth_convention(monkeypatch):
    monkeypatch.delenv("BLOOMBEE_EAGLE_DEPTH", raising=False)

    assert _eagle2_max_candidate_depth(4, explicit_tree_budget=True) == 6
    assert _eagle2_max_candidate_depth(5, explicit_tree_budget=True) == 6


def test_eagle_legacy_depth_remains_path_length(monkeypatch):
    monkeypatch.delenv("BLOOMBEE_EAGLE_DEPTH", raising=False)

    assert _eagle2_max_candidate_depth(5, explicit_tree_budget=False) == 5


def test_eagle_depth_env_controls_paper_budget_depth(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_EAGLE_DEPTH", "7")

    assert _eagle2_max_candidate_depth(4, explicit_tree_budget=True) == 8


def test_eagle_prefix_cache_is_isolated_per_batch_row():
    drafter = object.__new__(EAGLEDrafter)
    drafter.device = torch.device("cpu")
    drafter.dtype = torch.float32
    drafter.head_cfg = None
    drafter._prefix_states = {}
    step_calls = []

    def fake_step(hidden_states, input_ids, position_ids, past_key_values, attention_mask=None):
        step_calls.append((input_ids.clone(), position_ids.clone()))
        return hidden_states + input_ids.unsqueeze(-1).to(hidden_states.dtype), past_key_values

    drafter._step = fake_step

    h0 = torch.zeros(1, 2, 3)
    h1 = torch.ones(1, 2, 3)
    drafter._prefill_with_prefix(h0, torch.tensor([[11, 12]]), cache_key=0)
    drafter._prefill_with_prefix(h1, torch.tensor([[21, 22]]), cache_key=1)

    h0_extended = torch.zeros(1, 3, 3)
    drafter._prefill_with_prefix(h0_extended, torch.tensor([[11, 12, 13]]), cache_key=0)

    assert set(drafter._prefix_states) == {0, 1}
    assert drafter._prefix_states[0].cache_len == 3
    assert drafter._prefix_states[1].cache_len == 2
    assert step_calls[-1][0].tolist() == [[13]]
    assert step_calls[-1][1].tolist() == [[2]]


def test_eagle_tree_attention_mask_accepts_batched_masks():
    drafter = object.__new__(EAGLEDrafter)
    drafter.device = torch.device("cpu")

    mask = torch.tensor(
        [
            [[True, False], [True, True]],
            [[False, True], [True, False]],
        ]
    )

    additive = drafter._tree_attention_mask(mask, prefix_len=3, dtype=torch.float32)

    assert additive.shape == (2, 1, 2, 5)
    assert torch.all(additive[:, :, :, :3] == 0)
    assert additive[0, 0, 0, 3].item() == 0
    assert additive[0, 0, 0, 4].item() == torch.finfo(torch.float32).min
    assert additive[1, 0, 0, 3].item() == torch.finfo(torch.float32).min
    assert additive[1, 0, 0, 4].item() == 0


def test_eagle_build_trees_batches_prefix_rows():
    drafter = object.__new__(EAGLEDrafter)
    drafter.device = torch.device("cpu")
    drafter.dtype = torch.float32
    drafter._prefix_states = {}
    batched_calls = []

    def fake_prefill(prefix_hidden_states, shifted_input_ids, *, cache_key=0):
        return prefix_hidden_states[0, -1, :], object(), int(shifted_input_ids.shape[1])

    def fake_batched(*, jobs, max_candidate_depth, total_token, expansion_width):
        batched_calls.append((jobs, max_candidate_depth, total_token, expansion_width))
        return [
            SpeculativeTree(job.root_token, request_id=f"batched_{job.batch_index}")
            for job in jobs
        ]

    def fail_rowwise(**_):
        raise AssertionError("prefix rows should be expanded as one batched EAGLE call")

    drafter._prefill_with_prefix = fake_prefill
    drafter._build_trees_from_prefix_caches_batched = fake_batched
    drafter._build_tree_from_prefix_cache = fail_rowwise

    trees = drafter.build_trees_parallel(
        input_ids=torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]),
        seq_lengths=torch.tensor([4, 4]),
        prefix_hidden_states=torch.randn(2, 3, 4),
        prev_last_token=torch.tensor([4, 8]),
        beam_width=1,
        max_depth=5,
        tree_budget=5,
        topk_per_step=2,
    )

    assert len(batched_calls) == 1
    jobs = batched_calls[0][0]
    assert [job.batch_index for job in jobs] == [0, 1]
    assert [job.root_token for job in jobs] == [4, 8]
    assert [tree.root.token_id for tree in trees] == [4, 8]


def test_eagle_default_topk_uses_compact_latency_tree_for_draft5(monkeypatch):
    monkeypatch.delenv("BLOOMBEE_EAGLE_TOPK_PER_STEP", raising=False)

    assert _default_eagle_topk_per_step(11, explicit_tree_budget=True, do_sample=False) == 3
    assert _default_eagle_topk_per_step(16, explicit_tree_budget=True, do_sample=False) == 5
    assert _default_eagle_topk_per_step(21, explicit_tree_budget=True, do_sample=False) == 5


def test_eagle_default_topk_keeps_paper_tree_width(monkeypatch):
    monkeypatch.delenv("BLOOMBEE_EAGLE_TOPK_PER_STEP", raising=False)

    assert _default_eagle_topk_per_step(60, explicit_tree_budget=True, do_sample=False) == 10


def test_eagle_topk_env_override_wins(monkeypatch):
    monkeypatch.setenv("BLOOMBEE_EAGLE_TOPK_PER_STEP", "7")

    assert _default_eagle_topk_per_step(16, explicit_tree_budget=True, do_sample=False) == 7


def test_speculative_session_length_uses_current_tree_peak():
    assert (
        _speculative_session_max_length(
            prompt_len=23,
            max_new_tokens=32,
            beam_width=1,
            max_tree_depth=5,
            effective_tree_budget=10,
        )
        == 87
    )


def test_speculative_session_length_reserves_paper_tree_peak():
    assert (
        _speculative_session_max_length(
            prompt_len=23,
            max_new_tokens=32,
            beam_width=1,
            max_tree_depth=5,
            effective_tree_budget=59,
        )
        == 123
    )


def test_eagle_sampling_latency_tree_keeps_paper_topk(monkeypatch):
    monkeypatch.delenv("BLOOMBEE_EAGLE_DEPTH", raising=False)
    monkeypatch.delenv("BLOOMBEE_EAGLE_TOPK_PER_STEP", raising=False)

    drafter = object.__new__(EAGLEDrafter)
    drafter.device = torch.device("cpu")
    drafter.dtype = torch.float32
    drafter._prefix_states = {}
    calls = []

    def fake_prefill(prefix_hidden_states, shifted_input_ids, *, cache_key=0):
        return prefix_hidden_states[0, -1, :], object(), int(shifted_input_ids.shape[1])

    def fake_build(**kwargs):
        calls.append((
            kwargs["max_candidate_depth"],
            kwargs["total_token"],
            kwargs["expansion_width"],
        ))
        return SpeculativeTree(kwargs["root_tok"], request_id="sampling")

    drafter._prefill_with_prefix = fake_prefill
    drafter._build_tree_from_prefix_cache = fake_build

    drafter.build_trees_parallel(
        input_ids=torch.tensor([[1, 2, 3]]),
        seq_lengths=torch.tensor([3]),
        prefix_hidden_states=torch.randn(1, 2, 4),
        prev_last_token=torch.tensor([3]),
        beam_width=1,
        max_depth=5,
        tree_budget=15,
        do_sample=True,
    )

    assert calls == [(6, 16, 10)]


def test_eagle_greedy_latency_tree_uses_small_topk(monkeypatch):
    monkeypatch.delenv("BLOOMBEE_EAGLE_DEPTH", raising=False)
    monkeypatch.delenv("BLOOMBEE_EAGLE_TOPK_PER_STEP", raising=False)

    drafter = object.__new__(EAGLEDrafter)
    drafter.device = torch.device("cpu")
    drafter.dtype = torch.float32
    drafter._prefix_states = {}
    calls = []

    def fake_prefill(prefix_hidden_states, shifted_input_ids, *, cache_key=0):
        return prefix_hidden_states[0, -1, :], object(), int(shifted_input_ids.shape[1])

    def fake_build(**kwargs):
        calls.append((
            kwargs["max_candidate_depth"],
            kwargs["total_token"],
            kwargs["expansion_width"],
        ))
        return SpeculativeTree(kwargs["root_tok"], request_id="latency")

    drafter._prefill_with_prefix = fake_prefill
    drafter._build_tree_from_prefix_cache = fake_build

    drafter.build_trees_parallel(
        input_ids=torch.tensor([[1, 2, 3]]),
        seq_lengths=torch.tensor([3]),
        prefix_hidden_states=torch.randn(1, 2, 4),
        prev_last_token=torch.tensor([3]),
        beam_width=1,
        max_depth=5,
        tree_budget=10,
        do_sample=False,
    )

    assert calls == [(6, 11, 3)]


def test_eagle_paper_tree_default_uses_paper_topk(monkeypatch):
    monkeypatch.delenv("BLOOMBEE_EAGLE_DEPTH", raising=False)
    monkeypatch.delenv("BLOOMBEE_EAGLE_TOPK_PER_STEP", raising=False)

    drafter = object.__new__(EAGLEDrafter)
    drafter.device = torch.device("cpu")
    drafter.dtype = torch.float32
    drafter._prefix_states = {}
    calls = []

    def fake_prefill(prefix_hidden_states, shifted_input_ids, *, cache_key=0):
        return prefix_hidden_states[0, -1, :], object(), int(shifted_input_ids.shape[1])

    def fake_build(**kwargs):
        calls.append((
            kwargs["max_candidate_depth"],
            kwargs["total_token"],
            kwargs["expansion_width"],
        ))
        return SpeculativeTree(kwargs["root_tok"], request_id="paper")

    drafter._prefill_with_prefix = fake_prefill
    drafter._build_tree_from_prefix_cache = fake_build

    drafter.build_trees_parallel(
        input_ids=torch.tensor([[1, 2, 3]]),
        seq_lengths=torch.tensor([3]),
        prefix_hidden_states=torch.randn(1, 2, 4),
        prev_last_token=torch.tensor([3]),
        beam_width=1,
        max_depth=5,
        tree_budget=59,
        do_sample=False,
    )

    assert calls == [(6, 60, 10)]


def test_speculative_generate_passes_topk_per_step_to_eagle_drafter():
    model = object.__new__(DistributedLlamaForSpeculativeGeneration)
    input_ids = torch.tensor([[1, 10, 11]])
    captured = {}

    class FakeSession:
        position = 0

    class FakeDrafter:
        uses_eagle_hidden_states = True

        def build_trees_parallel(self, *args, **kwargs):
            captured["topk_per_step"] = kwargs.get("topk_per_step")
            captured["tree_budget"] = kwargs.get("tree_budget")
            return [SpeculativeTree(11, request_id="fake")]

    def fake_verify(**kwargs):
        batch_size = int(kwargs["input_ids"].shape[0])
        seq_len = int(kwargs["seq_lengths"].max().item())
        hidden = torch.zeros(batch_size, seq_len, 4)
        return (
            None,
            torch.tensor([[seq_len - 1]]),
            kwargs["past_key_values"],
            torch.tensor([[12]]),
            torch.zeros(batch_size, dtype=torch.long),
            hidden,
            torch.tensor([seq_len - 1]),
        )

    model._verify_trees_with_forward = fake_verify
    out = model._sample_with_session(
        input_ids=input_ids,
        drafter=FakeDrafter(),
        logits_processor=LogitsProcessorList(),
        stopping_criteria=StoppingCriteriaList(),
        generation_config=GenerationConfig(pad_token_id=0, eos_token_id=None),
        session=FakeSession(),
        streamer=None,
        beam_width=1,
        max_tree_depth=5,
        max_new_tokens=1,
        tree_budget=10,
        topk_per_step=7,
    )

    assert captured == {"tree_budget": 10, "topk_per_step": 7}
    assert out.tolist() == [[1, 10, 11, 12]]


def test_speculative_generate_prefers_attention_mask_for_initial_lengths():
    model = object.__new__(DistributedLlamaForSpeculativeGeneration)
    input_ids = torch.tensor([[1, 2, 11, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]])
    captured = {}

    class FakeSession:
        position = 0

    class FakeDrafter:
        uses_eagle_hidden_states = True

        def build_trees_parallel(self, _input_ids, seq_lengths, *args, **kwargs):
            captured["seq_lengths"] = seq_lengths.detach().cpu().tolist()
            return [SpeculativeTree(11, request_id="fake")]

    def fake_verify(**kwargs):
        batch_size = int(kwargs["input_ids"].shape[0])
        seq_len = int(kwargs["seq_lengths"].max().item())
        hidden = torch.zeros(batch_size, seq_len, 4)
        return (
            None,
            torch.tensor([[seq_len - 1]]),
            kwargs["past_key_values"],
            torch.tensor([[12]]),
            torch.zeros(batch_size, dtype=torch.long),
            hidden,
            torch.tensor([seq_len - 1]),
        )

    model._verify_trees_with_forward = fake_verify
    out = model._sample_with_session(
        input_ids=input_ids,
        drafter=FakeDrafter(),
        logits_processor=LogitsProcessorList(),
        stopping_criteria=StoppingCriteriaList(),
        generation_config=GenerationConfig(pad_token_id=0, eos_token_id=None),
        session=FakeSession(),
        streamer=None,
        beam_width=1,
        max_tree_depth=5,
        max_new_tokens=1,
        tree_budget=10,
        attention_mask=attention_mask,
    )

    assert captured == {"seq_lengths": [3]}
    assert out.tolist() == [[1, 2, 11, 12]]


def test_speculative_verify_can_use_drafter_lm_head_copy():
    class SlowHead(torch.nn.Module):
        def forward(self, hidden_states):
            raise AssertionError("drafter lm_head copy should be used when available")

    class DrafterWithHead:
        lm_head_weight = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        )
        lm_head_bias = torch.tensor([0.5, -0.5, 1.0])

    hidden = torch.tensor([[[2.0, 3.0]]])

    logits = _project_lm_head(SlowHead(), hidden, DrafterWithHead())

    assert logits.tolist() == [[[2.5, 2.5, 6.0]]]


def test_greedy_verify_projects_only_accepted_path_and_bonus():
    class CountingHead:
        def __init__(self):
            self.weight = torch.zeros(50, 4)
            self.weight[20, 0] = 10.0
            self.weight[30, 1] = 10.0
            self.weight[40, 2] = 10.0
            self.tokens_projected = 0

        def __call__(self, hidden_states):
            self.tokens_projected += int(hidden_states.shape[-2])
            return torch.matmul(hidden_states, self.weight.t())

    tree = SpeculativeTree(root_token=10, request_id="row0")
    node20 = tree.root.add_child(20, 0.9)
    tree.root.add_child(99, 0.1)
    node20.add_child(30, 0.8)
    tree.total_nodes = 4
    tree.max_depth = 2
    linearize_tree_with_positions(tree)

    model = object.__new__(DistributedLlamaForSpeculativeGeneration)
    model.lm_head = CountingHead()
    hidden_states = torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0],
          [0.0, 1.0, 0.0, 0.0],
          [0.0, 0.0, 1.0, 0.0],
          [0.0, 0.0, 0.0, 1.0]]]
    )

    verified, positions, bonus, valid, final_positions = model._extract_greedy_verified_paths_from_hidden(
        hidden_states=hidden_states,
        trees=[tree],
        input_ids=torch.tensor([[7, 8, 10]]),
        logits_processor=[],
        tree_len=3,
        seq_lengths=torch.tensor([3]),
        is_first_iteration=False,
        drafter=None,
    )

    assert verified.tolist() == [[20, 30]]
    assert positions.tolist() == [[2, 3, 4]]
    assert bonus.tolist() == [[40]]
    assert valid.tolist() == [2]
    assert final_positions.tolist() == [2]
    assert model.lm_head.tokens_projected == 3


def test_eagle_prefix_hidden_appends_first_round_accepted_hiddens():
    model = object.__new__(DistributedLlamaForSpeculativeGeneration)
    # verify_hidden_states layout on the first speculative round:
    # prompt positions [0, 1, 2], then accepted draft nodes at [3, 4].
    verify_hidden = torch.arange(1 * 5 * 2, dtype=torch.float32).view(1, 5, 2)
    kv_positions = torch.tensor([[2, 3, 4]])

    prefix = model._update_eagle_prefix_hidden_states(
        prefix_hidden_states=None,
        verify_hidden_states=verify_hidden,
        kv_cache_position_ids=kv_positions,
        old_seq_lengths=torch.tensor([3]),
        is_first_iteration=True,
    )

    assert prefix.shape == (1, 5, 2)
    torch.testing.assert_close(prefix[0], verify_hidden[0])


def test_speculative_generation_honors_do_sample_kwarg():
    cfg = GenerationConfig(do_sample=True, temperature=0.6, pad_token_id=0)
    kwargs = {"do_sample": False, "temperature": 1.0, "pad_token_id": 2}

    merged = _merge_generation_config_kwargs(cfg, kwargs)

    assert merged.do_sample is False
    assert merged.temperature == 1.0
    assert merged.pad_token_id == 2
    assert cfg.do_sample is True
    assert kwargs == {}


def test_speculative_generation_reads_generation_config_eos_ids():
    assert _eos_token_ids(GenerationConfig(eos_token_id=None)) == ()
    assert _eos_token_ids(GenerationConfig(eos_token_id=2)) == (2,)
    assert _eos_token_ids(GenerationConfig(eos_token_id=[2, 32000])) == (2, 32000)


def test_speculative_generation_caps_last_step_to_max_new_tokens():
    valid_lengths = torch.tensor([5, 2, 0])
    seq_lengths = torch.tensor([14, 10, 15])

    capped, append_llm = _cap_valid_lengths_to_remaining(
        valid_lengths=valid_lengths,
        seq_lengths=seq_lengths,
        initial_len=10,
        max_new_tokens=5,
    )

    assert capped.tolist() == [1, 2, 0]
    assert append_llm.tolist() == [0, 1, 0]


def test_speculative_generation_caps_variable_length_batch_per_row():
    valid_lengths = torch.tensor([5, 0, 5])
    seq_lengths = torch.tensor([9, 14, 20])
    initial_lengths = torch.tensor([4, 10, 20])

    capped, append_llm = _cap_valid_lengths_to_remaining(
        valid_lengths=valid_lengths,
        seq_lengths=seq_lengths,
        initial_len=initial_lengths,
        max_new_tokens=5,
    )

    assert capped.tolist() == [0, 0, 5]
    assert append_llm.tolist() == [0, 1, 0]


def test_speculative_fallback_masks_right_padded_batch():
    mask = _attention_mask_from_seq_lengths(torch.tensor([2, 4]), max_seq_len=4)

    assert mask.tolist() == [
        [True, True, False, False],
        [True, True, True, True],
    ]
