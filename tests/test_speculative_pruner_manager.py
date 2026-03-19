import logging

import torch

from bloombee.server.speculative_pruner import pruner_manager as pruner_manager_module


class _DummyPruner:
    def get_metrics(self):
        return {}


def test_missing_lm_head_weights_do_not_crash(monkeypatch, caplog):
    monkeypatch.setattr(
        pruner_manager_module.SpeculativePrunerFactory,
        "create_pruner",
        lambda *args, **kwargs: _DummyPruner(),
    )

    class _MissingTrainer:
        def __init__(self, *args, **kwargs):
            raise FileNotFoundError("/tmp/data/llama_weights/llama-7b-np")

    monkeypatch.setattr(pruner_manager_module, "LM_head_trainer", _MissingTrainer)

    manager = pruner_manager_module.SpeculativePrunerManager(hidden_size=4, vocab_size=8, device="cpu")

    caplog.set_level(logging.WARNING)
    manager.train_lm_head(torch.zeros(1, 1, 4), torch.zeros(1, 1, 4))

    assert manager.lm_head_trainer is None
    assert manager._lm_head_trainer_unavailable is True
    assert manager.iteration == 0
    assert "Disabling auxiliary LM-head trainer" in caplog.text

    caplog.clear()
    manager.train_lm_head(torch.zeros(1, 1, 4), torch.zeros(1, 1, 4))
    assert caplog.text == ""
