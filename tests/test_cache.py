import asyncio
import multiprocessing as mp
import random
import time
from typing import Optional

import pytest
import pytest_asyncio  # make sure the module exists; otherwise the test will be skipped
import torch
from hivemind import TensorDescriptor

from types import SimpleNamespace
from bloombee.flexgen_utils.pytorch_backend import TorchDevice
from bloombee.flexgen_utils.ExecutionEnv import ExecutionEnv
from bloombee.flexgen_utils.policy import Policy
from bloombee.flexgen_utils.compression import CompressionConfig

from bloombee.server.memory_cache import AllocationFailed, MemoryCache
from bloombee.server.memory_cache_manager import KVCacheManager

from bloombee.utils.misc import get_size_in_bytes

def _make_tensor_descriptor(num_bytes: int, dtype: Optional[torch.dtype] = None):
    if dtype is None:
        dtype = random.choice((torch.int64, torch.int8, torch.uint8, torch.float32, torch.bfloat16, torch.bool))
    elem_size_bytes = get_size_in_bytes(dtype)
    descr = TensorDescriptor.from_tensor(torch.empty((num_bytes // elem_size_bytes,), dtype=dtype))
    return descr

def _make_token_descriptor(n_tokens: int):
    return TensorDescriptor.from_tensor(torch.empty((1, 1, n_tokens), dtype=torch.float16))

def _make_kv_cache_manager(max_tokens: int, max_alloc_timeout=None):
    cpu_device = TorchDevice("cpu")
    # Construct env directly — avoids TorchDisk which spawns 4 copy threads
    env = ExecutionEnv(gpu=cpu_device, cpu=cpu_device, disk=None, mixed=None)

    # CPU-only, no CUDA required
    policy = Policy(
        1, 1,        # gpu_batch_size, num_gpu_batches
        100, 0,      # w_gpu_percent, w_cpu_percent
        0, 100,      # cache_gpu_percent=0, cache_cpu_percent=100  (all cache on CPU)
        100, 0,      # act_gpu_percent, act_cpu_percent
        overlap=False, sep_layer=True, pin_weight=False,
        cpu_cache_compute=False, attn_sparsity=1.0,
        compress_weight=False,
        comp_weight_config=CompressionConfig(num_bits=4, group_size=64, group_dim=0, symmetric=False),
        compress_cache=False,
        comp_cache_config=CompressionConfig(num_bits=4, group_size=64, group_dim=2, symmetric=False),
    )
    block_config = SimpleNamespace(
        num_attention_heads=4,
        hidden_size=32,
        num_key_value_groups=1,
    )

    return KVCacheManager(max_tokens, max_alloc_timeout, policy, env, block_config)


@pytest.mark.asyncio
async def test_cache_timeout():
    manager = _make_kv_cache_manager(max_tokens=1024, max_alloc_timeout=0.5)
    manager.runtime_pid += 1  # pretend we're another process
    async with manager.allocate_cache(_make_token_descriptor(768), timeout=0):
        pass

    async with manager.allocate_cache(_make_token_descriptor(100), timeout=999):
        async with manager.allocate_cache(_make_token_descriptor(512), timeout=0):
            async with manager.allocate_cache(_make_token_descriptor(128), _make_token_descriptor(32), timeout=1):
                t_start = time.perf_counter()
                with pytest.raises(AllocationFailed):
                    async with manager.allocate_cache(_make_token_descriptor(768), timeout=0.1):
                        pass
                assert 0.1 < time.perf_counter() - t_start < 0.2, "wait time exceeds alloc timeout"
                async with manager.allocate_cache(_make_token_descriptor(128), timeout=float("inf")):
                    pass

                t_start = time.perf_counter()
                with pytest.raises(AllocationFailed):
                    async with manager.allocate_cache(_make_token_descriptor(384), timeout=1.0):  # exceeds max timeout
                        pass
                assert 0.5 < time.perf_counter() - t_start < 0.6, "wait time exceeds max alloc timeout"

            # test memory allocation when another task frees the memory
            async def _klog_the_cache():
                async with manager.allocate_cache(_make_token_descriptor(512), timeout=0.2):
                    pass

            large_alloc_task = asyncio.create_task(_klog_the_cache())

            t_start = time.perf_counter()
            await asyncio.sleep(0.05)  # wait for large alloc to enqueue
            async with manager.allocate_cache(_make_token_descriptor(128), timeout=float("inf")):  # exceeds max timeout
                pass  # this memory should allocate once the background task clears the queue
            assert 0.2 < time.perf_counter() - t_start < 0.3, "memory should be allocated after background task clears"
            with pytest.raises(AllocationFailed):
                await large_alloc_task

            # test that zero-timeout allocation fails instantaneously even if someone else is awaiting alloc
            large_alloc_task = asyncio.create_task(_klog_the_cache())
            t_start = time.perf_counter()
            await asyncio.sleep(0.05)  # wait for large alloc to enqueue
            with pytest.raises(AllocationFailed):
                async with manager.allocate_cache(_make_token_descriptor(512), timeout=0):
                    pass  # this memory should allocate once the background task clears the queue
            assert time.perf_counter() - t_start < 0.1, "zero-timeout task should fail (or succeed) instantaneously"
            with pytest.raises(AllocationFailed):
                await large_alloc_task


@pytest.mark.asyncio
async def test_unlimited_timeout():
    manager = _make_kv_cache_manager(max_tokens=1024)
    manager.runtime_pid += 1  # pretend we're another process
    t_start = time.perf_counter()

    async def _klog_the_cache():
        async with manager.allocate_cache(_make_token_descriptor(512), timeout=0.2):
            await asyncio.sleep(0.5)

    alloc_task = asyncio.create_task(_klog_the_cache())
    await asyncio.sleep(0.1)
    async with manager.allocate_cache(_make_token_descriptor(768), timeout=float("inf")):
        await alloc_task
    assert 0.5 < time.perf_counter() - t_start < 0.6, "memory should be allocated after background task clears"


@pytest.mark.asyncio
async def test_cache_usage():
    manager = _make_kv_cache_manager(max_tokens=2048)
    # main process is the "runtime" (calls use_cache); fork processes are "handlers" (call allocate_cache).
    # Do NOT bump runtime_pid: fork children have a different PID automatically.
    alloc_event, dealloc_a_event, dealloc_bcd_event, dealloc_e_event, dealloc_f_event = (mp.Event() for _ in range(5))
    pipe_receiver, pipe_sender = mp.Pipe(duplex=False)
    with pytest.raises(AssertionError):
        async with manager.allocate_cache(_make_token_descriptor(128), timeout=1):  # No specific reason to use 128
            pass  # fails because cache must be allocated from another process

    async def _allocate_and_wait(dealloc_event, *descrs, timeout=None):
        loop = asyncio.get_event_loop()
        async with manager.allocate_cache(*descrs, timeout=timeout) as handles:
            pipe_sender.send(handles)
            await loop.run_in_executor(None, dealloc_event.wait)

    async def _allocate_af():
        alloc_event.wait()
        await asyncio.create_task(_allocate_and_wait(dealloc_a_event, _make_token_descriptor(768)))
        await asyncio.create_task(_allocate_and_wait(dealloc_f_event, _make_token_descriptor(1792)))  # klogs the cache

    alloc_process1 = mp.context.ForkProcess(target=lambda: asyncio.run(_allocate_af()), daemon=True)
    alloc_process1.start()

    async def _allocate_bcde():
        alloc_event.wait()
        await asyncio.sleep(0.1)  # ensure that the other tensor is always allocated (and sent through pipe) first
        allocate_bcd_task = asyncio.create_task(_allocate_and_wait(
            dealloc_bcd_event, _make_token_descriptor(100), _make_token_descriptor(512), _make_token_descriptor(0)
        ))
        allocate_e_task = asyncio.create_task(_allocate_and_wait(
            dealloc_e_event, _make_token_descriptor(1536)
        ))  # doesn't fit
        await asyncio.wait({allocate_e_task, allocate_bcd_task}, return_when=asyncio.ALL_COMPLETED)

    alloc_process2 = mp.context.ForkProcess(target=lambda: asyncio.run(_allocate_bcde()), daemon=True)
    alloc_process2.start()
    assert manager.current_size_tokens == 0
    alloc_event.set()
    (handle_a,) = pipe_receiver.recv()

    handle_b, handle_c, handle_d = pipe_receiver.recv()

    with manager.use_cache(handle_a) as tensors:
        (k_a, v_a), = tensors
        assert k_a.shape == (768, 4, 8)

    with manager.use_cache(handle_a, handle_b, handle_d) as tensors:
        (k_a, v_a), (k_b, v_b), (k_d, v_d) = tensors
        assert k_b.shape == (100, 4, 8)
    assert manager.current_size_tokens == 1280 # a=768 + bcd=max(100,512,0)=512

    dealloc_bcd_event.set()
    await asyncio.sleep(0.1)
    assert manager.current_size_tokens == 768  # only tensor a should be allocated
    # Tests would hang, pending Fix in memory_cache.py
    # with pytest.raises(KeyError):
    #     with manager.use_cache(handle_a, handle_b):
    #         pass  # one of handles (c) is deallocated
    # with manager.raises(KeyError):
    #     with cache.use_cache(handle_d):
    #         pass  # handle_d is deallocated correctly, even though it is never used
    with manager.use_cache(handle_a) as tensors:
        (k_a, v_a), = tensors
        assert k_a.shape == (768, 4, 8)

    dealloc_a_event.set()
    (handle_e,) = pipe_receiver.recv()  # e can finally be allocated
    await asyncio.sleep(0.1)
    assert manager.current_size_tokens == 1536  # tensor e should finally be able to allocate

    # Tests would hang, pending Fix in memory_cache.py
    # with pytest.raises(KeyError):
    #     with manager.use_cache(handle_a):
    #         pass  # tensor a is no longer allocated
    with manager.use_cache(handle_e) as tensors:
        (k_e, v_e), = tensors
        assert k_e.shape == (1536, 4, 8)

    dealloc_e_event.set()
    await asyncio.sleep(0.1)
    assert manager.current_size_tokens == 1792  # only tensor f is still allocated
    dealloc_f_event.set()

    alloc_process1.join()
    alloc_process2.join()
    await asyncio.sleep(0.1)
    assert manager.current_size_tokens == 0
    assert alloc_process1.exitcode == 0, "allocation process 1 failed or did not finish, see stderr for details"
    assert alloc_process2.exitcode == 0, "allocation process 2 failed or did not finish, see stderr for details"
