import asyncio
import gc
from contextlib import contextmanager, suppress

import psutil
import pytest
from hivemind.utils.crypto import RSAPrivateKey
from hivemind.utils.logging import get_logger
from hivemind.utils.mpfuture import MPFuture

logger = get_logger(__name__)


@pytest.fixture
def make_cpu_cache_manager():
    """Real cache managers without CUDA or disk copy threads."""
    from bloombee.flexgen_utils.compression import CompressionConfig
    from bloombee.flexgen_utils.ExecutionEnv import ExecutionEnv
    from bloombee.flexgen_utils.policy import Policy
    from bloombee.flexgen_utils.pytorch_backend import TorchDevice, fix_recursive_import
    from bloombee.server.memory_cache_manager import KVCacheManager

    managers = []

    def make(config):
        fix_recursive_import()
        device = TorchDevice("cpu")
        compression = CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False)
        policy = Policy(
            gpu_batch_size=1,
            num_gpu_batches=1,
            w_gpu_percent=100,
            w_cpu_percent=0,
            cache_gpu_percent=0,
            cache_cpu_percent=100,
            act_gpu_percent=100,
            act_cpu_percent=0,
            overlap=False,
            sep_layer=True,
            pin_weight=False,
            cpu_cache_compute=False,
            attn_sparsity=1.0,
            compress_weight=False,
            comp_weight_config=compression,
            compress_cache=False,
            comp_cache_config=compression,
        )
        manager = KVCacheManager(4096, None, policy, ExecutionEnv(gpu=device, cpu=device), config)
        managers.append(manager)
        return manager

    yield make
    for manager in managers:
        manager.wait_for_pending_reorder()
        manager._reorder_executor.shutdown(wait=True)


@pytest.fixture
def local_remote_layers():
    """Replace network transport while retaining the production generation stack."""
    import torch

    class LocalRemoteLayers(torch.nn.Module):
        def __init__(self, config, dht=None):
            super().__init__()
            self.active_session = None

        @property
        def position(self):
            return self.active_session.position if self.active_session is not None else 0

        @contextmanager
        def use_session(self, session):
            previous = self.active_session
            self.active_session = session
            try:
                yield session
            finally:
                self.active_session = previous

        def forward(self, hidden_states, **kwargs):
            self.active_session.position += hidden_states.shape[1]
            return hidden_states

    return LocalRemoteLayers


@pytest.fixture
def event_loop():
    """
    This overrides the ``event_loop`` fixture from pytest-asyncio
    (e.g. to make it compatible with ``asyncio.subprocess``).

    This fixture is identical to the original one but does not call ``loop.close()`` in the end.
    Indeed, at this point, the loop is already stopped (i.e. next tests are free to create new loops).
    However, finalizers of objects created in the current test may reference the current loop and fail if it is closed.
    For example, this happens while using ``asyncio.subprocess`` (the ``asyncio.subprocess.Process`` finalizer
    fails if the loop is closed, but works if the loop is only stopped).
    """

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # Earlier synchronous tests may clear the thread's current loop.
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    yield loop


@pytest.fixture(autouse=True, scope="session")
def cleanup_children():
    yield

    with RSAPrivateKey._process_wide_key_lock:
        RSAPrivateKey._process_wide_key = None

    gc.collect()  # Call .__del__() for removed objects

    children = psutil.Process().children(recursive=True)
    if children:
        logger.info(f"Cleaning up {len(children)} leftover child processes")
        for child in children:
            with suppress(psutil.NoSuchProcess):
                child.terminate()
        psutil.wait_procs(children, timeout=1)
        for child in children:
            with suppress(psutil.NoSuchProcess):
                child.kill()

    MPFuture.reset_backend()
