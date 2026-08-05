import os

import pytest

# Swarm-dependent tests import this module for INITIAL_PEERS/MODEL_NAME.
# Skip (not error) when the env is absent so a plain `pytest tests/` run
# collects cleanly and only exercises the local unit tests.
INITIAL_PEERS = os.environ.get("INITIAL_PEERS")
if not INITIAL_PEERS:
    pytest.skip(
        "Set INITIAL_PEERS (and MODEL_NAME) to run swarm-dependent tests",
        allow_module_level=True,
    )
INITIAL_PEERS = INITIAL_PEERS.split()


MODEL_NAME = os.environ.get("MODEL_NAME")
if not MODEL_NAME:
    pytest.skip(
        "Set MODEL_NAME to run swarm-dependent tests",
        allow_module_level=True,
    )

REF_NAME = os.environ.get("REF_NAME")

ADAPTER_NAME = os.environ.get("ADAPTER_NAME")
