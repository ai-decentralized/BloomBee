"""
Utils for fetching pretrained model parts. Currently, this relies on huggingface transformers' from_pretrained code.
If necessary, one can rewrite this to implement a different behavior, such as:
 - loading files from a local data source (e.g. S3)
 - load files via BitTorrent ( https://pypi.org/project/libtorrent/ ) or IPFS( https://docs.ipfs.io/how-to )
 - fetch the weights over IPoAC, using a fleet of trained pigeons ( http://www.faqs.org/rfcs/rfc1149.html )

"""
import json
import re
import time
from collections import defaultdict
from contextlib import suppress
from typing import Dict, Optional, Sequence, Union

import safetensors
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from hivemind.utils.logging import get_logger
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from huggingface_hub.utils import EntryNotFoundError
from transformers import PretrainedConfig, PreTrainedModel

from bloombee.utils.hf_compat import get_file_from_repo

from bloombee.constants import DTYPE_MAP
from bloombee.models.bloom.block import WrappedBloomBlock
from bloombee.models.deepseekv3.block import WrappedDeepseekV3Block
from bloombee.models.mixtral import WrappedMixtralBlock
from bloombee.models.nemotron_h.block import WrappedNemotronHBlock
from bloombee.models.falcon.block import WrappedFalconBlock
from bloombee.models.gemma4.block import WrappedGemma4Block
from bloombee.models.glm4_moe_lite.block import WrappedGlm4MoeLiteBlock
from bloombee.models.qwen3.block import WrappedQwen3Block
from bloombee.server.block_utils import get_model_block, resolve_block_dtype
from bloombee.utils.auto_config import AutoDistributedConfig
from bloombee.utils.disk_cache import DEFAULT_CACHE_DIR, allow_cache_reads, allow_cache_writes, free_disk_space_for
from bloombee.utils.hf_auth import always_needs_auth

from bloombee.flexgen_utils.llama_config import LlamaConfig, get_llama_config, download_llama_weights
from bloombee.flexgen_utils.ExecutionEnv import ExecutionEnv
from bloombee.flexgen_utils.compression import CompressionConfig
from bloombee.flexgen_utils.policy import Policy
from bloombee.flexgen_utils.pytorch_backend import fix_recursive_import, TorchTensor
from bloombee.flexgen_utils.utils import ValueHolder, array_1d

from bloombee.models.llama.flex_llama import load_weights_from_pytorch_model
import os





DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes
logger = get_logger(__name__)


def load_pretrained_block(
    model_name: str,
    block_index: int,
    env: Optional[ExecutionEnv] = None,
    policy: Optional[Policy] = None,
    weight_home: Optional[array_1d] = None,
    path: Optional[str] = None,
    *,
    config: Optional[PretrainedConfig] = None,
    torch_dtype: Union[torch.dtype, str] = "auto",
    revision: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
    cache_dir: Optional[str] = None,
    max_disk_space: Optional[int] = None,
    tensor_parallel_devices: Optional[Sequence[torch.device]] = None,
) -> nn.Module:
    if config is None:
        config = AutoDistributedConfig.from_pretrained(model_name, use_auth_token=token)
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR

    assert torch_dtype in DTYPE_MAP.values(), f"torch_dtype must be one of {list(DTYPE_MAP.values())}"
    torch_dtype = resolve_block_dtype(config, torch_dtype)

    # transformers 5.x returns configs with an empty _name_or_path; FlexGen's
    # weight resolution reads it to locate/convert weights, so thread the real
    # model name back in for every load path (not just the non-TP one).
    if model_name and not getattr(config, "_name_or_path", ""):
        try:
            config._name_or_path = model_name
        except Exception:
            pass
    try:
        config._bloombee_revision = revision
        config._bloombee_token = token
    except Exception:
        pass

    use_native_flexgen_llama_tp = (
        getattr(config, "model_type", None) == "llama"
        and tensor_parallel_devices is not None
        and len(tensor_parallel_devices) > 1
    )

    # Determine if this is a FlexGen-managed model (Llama) or a standard HF model (Bloom, Falcon, Mixtral, Qwen3, Gemma4)
    _is_hf_model = config.block_class in (
        WrappedBloomBlock,
        WrappedFalconBlock,
        WrappedMixtralBlock,
        WrappedQwen3Block,
        WrappedGemma4Block,
        WrappedDeepseekV3Block,
        WrappedNemotronHBlock,
        WrappedGlm4MoeLiteBlock,
    )

    needs_flexgen = (not _is_hf_model) and getattr(config, "model_type", None) != "gpt_oss"
    if needs_flexgen and None in (env, policy, weight_home, path):
        raise TypeError("LLaMA FlexGen load_pretrained_block requires env, policy, weight_home, and path")

    if use_native_flexgen_llama_tp:
        with init_empty_weights():
            logger.info(
                "Preparing FlexGen-native LLaMA TP block template for layer %s across %s devices",
                block_index,
                len(tensor_parallel_devices),
            )
            block = config.block_class(
                config,
                block_index,
                env,
                policy,
                weight_home,
                path,
                skip_init_weights=True,
            )
    elif _is_hf_model or config.model_type == "gpt_oss":
        # For HF-based models: create block normally (weights on CPU) then load state dict
        block = get_model_block(config, env, policy, weight_home, path, layer_idx=block_index)
        block = _load_hf_block_weights(
            block, model_name, block_index, config,
            revision=revision,
            token=token, cache_dir=cache_dir, max_disk_space=max_disk_space, torch_dtype=torch_dtype,
        )
    else:
        # For FlexGen-based models (Llama): weights are managed by FlexGen at runtime.
        # FlexGen's OptimizedLlamaDecoderLayer.init_weight() reads _name_or_path off
        # the config to decide between local conversion and the huggyllama download
        # fallback. For local directories, prefer the path the user passed to
        # from_pretrained over a stale _name_or_path saved inside config.json.
        if os.path.isdir(model_name):
            try:
                config._name_or_path = model_name
            except Exception:
                pass
        elif not getattr(config, "_name_or_path", "") and model_name:
            try:
                config._name_or_path = model_name
            except Exception:
                pass
        try:
            config._bloombee_revision = revision
            config._bloombee_token = token
        except Exception:
            pass
        with init_empty_weights():
            logger.debug('load_pretrained_block: init_empty_weights()')
            block = get_model_block(config, env, policy, weight_home, path, layer_idx=block_index)

    return block


_DEEPSEEKV3_LEGACY_EXPERT_KEY_RE = re.compile(r"^(.*mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$")
_MIXTRAL_LEGACY_EXPERT_KEY_RE = re.compile(
    r"^(.*)block_sparse_moe\.experts\.(\d+)\.(w1|w2|w3)\.weight$"
)
_MIXTRAL_LEGACY_GATE_KEY_RE = re.compile(r"^(.*)block_sparse_moe\.gate\.(weight|bias)$")


def _remap_deepseekv3_expert_state_dict(state_dict: "StateDict") -> "StateDict":
    """Merge DeepSeek-V3's legacy per-expert weight layout into the batched
    gate_up_proj/down_proj tensors that the installed transformers' DeepseekV3Experts
    module actually has as parameters.

    Some DeepSeek-V3 checkpoints (e.g. unsloth/DeepSeek-V3-bf16) still ship each routed
    expert as three separate nn.Linear-style tensors:
        mlp.experts.{i}.gate_proj.weight  [intermediate, hidden]
        mlp.experts.{i}.up_proj.weight    [intermediate, hidden]
        mlp.experts.{i}.down_proj.weight  [hidden, intermediate]
    but transformers' DeepseekV3Experts module (built by WrappedDeepseekV3Block) stores
    all experts as two 3D tensors instead:
        mlp.experts.gate_up_proj  [num_experts, 2*intermediate, hidden]
        mlp.experts.down_proj     [num_experts, hidden, intermediate]
    Feeding the old-format keys straight into `block.load_state_dict(..., strict=False)`
    silently no-ops on the name mismatch, leaving gate_up_proj/down_proj at their
    uninitialized values -- i.e. the routed experts (>95% of the model's parameters)
    never receive their trained weights while every other tensor loads fine.

    The merge order (gate then up, concatenated along the output-feature dim; experts
    stacked in index order) mirrors transformers' own conversion_mapping.py entry for
    "qwen2_moe", which deepseek_v3 is aliased to, so this matches upstream exactly.

    No-ops (returns state_dict unchanged) if no legacy per-expert keys are present --
    i.e. dense blocks, and checkpoints that already ship the batched format.

    Memory note: this fills the output gate_up_proj/down_proj tensors in place, one
    expert at a time, dropping each source tensor as soon as it's copied in (rather
    than `torch.stack`-ing all experts into intermediate tensors and `torch.cat`-ing
    those). With 256 experts per MoE layer, materializing gate_stack/up_stack/down_stack
    *and* the original per-expert tensors *and* the final concatenated result all at
    once roughly triples peak memory for that layer; this keeps peak at ~1x the raw
    expert-weight size instead.
    """
    per_expert = defaultdict(dict)  # {(prefix, expert_idx): {"gate_proj": t, "up_proj": t, "down_proj": t}}
    for key in list(state_dict.keys()):
        m = _DEEPSEEKV3_LEGACY_EXPERT_KEY_RE.match(key)
        if m is None:
            continue
        prefix, expert_idx, proj = m.group(1), int(m.group(2)), m.group(3)
        per_expert[(prefix, expert_idx)][proj] = state_dict.pop(key)

    if not per_expert:
        return state_dict

    by_prefix = defaultdict(dict)
    for (prefix, expert_idx), tensors in per_expert.items():
        by_prefix[prefix][expert_idx] = tensors
    per_expert.clear()  # drop the flat view's refs -- by_prefix now owns the only ones,
    # so popping an expert out of experts_by_idx below actually frees its tensors.

    for prefix, experts_by_idx in by_prefix.items():
        num_experts = max(experts_by_idx) + 1
        missing = [i for i in range(num_experts) if i not in experts_by_idx]
        if missing:
            raise ValueError(
                f"{prefix}: missing expert indices {missing} while remapping legacy weight layout "
                f"(found {sorted(experts_by_idx)})"
            )

        sample_gate = experts_by_idx[0]["gate_proj"]
        intermediate, hidden = sample_gate.shape
        dtype = sample_gate.dtype
        gate_up_proj = torch.empty((num_experts, 2 * intermediate, hidden), dtype=dtype)
        down_proj = torch.empty((num_experts, hidden, intermediate), dtype=dtype)

        for i in range(num_experts):
            tensors = experts_by_idx.pop(i)  # last ref to this expert's raw tensors
            gate_up_proj[i, :intermediate] = tensors.pop("gate_proj")
            gate_up_proj[i, intermediate:] = tensors.pop("up_proj")
            down_proj[i] = tensors.pop("down_proj")
            # `tensors` is now empty and falls out of scope here, so each expert's
            # three source tensors are freed immediately after being copied in,
            # instead of staying alive until the whole layer is done.

        state_dict[f"{prefix}.gate_up_proj"] = gate_up_proj
        state_dict[f"{prefix}.down_proj"] = down_proj

    return state_dict


def _remap_mixtral_expert_state_dict(state_dict: "StateDict") -> "StateDict":
    """Map Hub Mixtral per-expert keys onto Transformers 5 packed experts.

    Official checkpoints store:
        block_sparse_moe.gate.weight
        block_sparse_moe.experts.{i}.w1.weight  [intermediate, hidden]  (gate)
        block_sparse_moe.experts.{i}.w3.weight  [intermediate, hidden]  (up)
        block_sparse_moe.experts.{i}.w2.weight  [hidden, intermediate]  (down)
    Transformers 5 ``MixtralDecoderLayer`` uses:
        mlp.gate.weight
        mlp.experts.gate_up_proj  [E, 2*intermediate, hidden]  (w1 then w3)
        mlp.experts.down_proj     [E, hidden, intermediate]    (w2)
    """
    per_expert = defaultdict(dict)
    for key in list(state_dict.keys()):
        m = _MIXTRAL_LEGACY_EXPERT_KEY_RE.match(key)
        if m is None:
            continue
        stem, expert_idx, proj = m.group(1), int(m.group(2)), m.group(3)
        per_expert[(stem, expert_idx)][proj] = state_dict.pop(key)

    for key in list(state_dict.keys()):
        m = _MIXTRAL_LEGACY_GATE_KEY_RE.match(key)
        if m is None:
            continue
        stem, leaf = m.group(1), m.group(2)
        state_dict[f"{stem}mlp.gate.{leaf}"] = state_dict.pop(key)

    if not per_expert:
        return state_dict

    by_stem = defaultdict(dict)
    for (stem, expert_idx), tensors in per_expert.items():
        by_stem[stem][expert_idx] = tensors
    per_expert.clear()

    for stem, experts_by_idx in by_stem.items():
        num_experts = max(experts_by_idx) + 1
        missing = [i for i in range(num_experts) if i not in experts_by_idx]
        if missing:
            raise ValueError(
                f"{stem}block_sparse_moe: missing expert indices {missing} while remapping "
                f"legacy Mixtral weights (found {sorted(experts_by_idx)})"
            )
        sample = experts_by_idx[0].get("w1")
        if sample is None:
            raise ValueError(
                f"{stem}block_sparse_moe.experts.0: missing w1 while remapping legacy Mixtral weights"
            )
        intermediate, hidden = sample.shape
        dtype = sample.dtype
        gate_up_proj = torch.empty((num_experts, 2 * intermediate, hidden), dtype=dtype)
        down_proj = torch.empty((num_experts, hidden, intermediate), dtype=dtype)
        required = ("w1", "w2", "w3")
        for i in range(num_experts):
            tensors = experts_by_idx.pop(i)
            missing_proj = [name for name in required if name not in tensors]
            if missing_proj:
                raise ValueError(
                    f"{stem}block_sparse_moe.experts.{i}: missing {missing_proj} "
                    f"while remapping legacy Mixtral weights"
                )
            w1 = tensors.pop("w1")
            w3 = tensors.pop("w3")
            w2 = tensors.pop("w2")
            if tuple(w1.shape) != (intermediate, hidden) or tuple(w3.shape) != (intermediate, hidden):
                raise ValueError(
                    f"{stem}block_sparse_moe.experts.{i}: expected w1/w3 "
                    f"{(intermediate, hidden)}, got w1={tuple(w1.shape)} w3={tuple(w3.shape)}"
                )
            if tuple(w2.shape) != (hidden, intermediate):
                raise ValueError(
                    f"{stem}block_sparse_moe.experts.{i}: expected w2 "
                    f"{(hidden, intermediate)}, got {tuple(w2.shape)}"
                )
            gate_up_proj[i, :intermediate] = w1
            gate_up_proj[i, intermediate:] = w3
            down_proj[i] = w2
        prefix = f"{stem}mlp.experts"
        state_dict[f"{prefix}.gate_up_proj"] = gate_up_proj
        state_dict[f"{prefix}.down_proj"] = down_proj

    return state_dict


def _load_hf_block_weights(
    block: nn.Module,
    model_name: str,
    block_index: int,
    config: PretrainedConfig,
    *,
    revision: Optional[str] = None,
    token: Optional[Union[str, bool]],
    cache_dir: str,
    max_disk_space: Optional[int],
    torch_dtype: torch.dtype,
) -> nn.Module:
    """Load HF state dict weights into a block for non-FlexGen models (Falcon, Mixtral)."""
    block_prefix = f"{config.block_prefix}.{block_index}."
    logger.info(f"Loading HF weights for {block_prefix} from {model_name} revision={revision}")
    state_dict = _load_state_dict_from_repo(
        model_name,
        block_prefix,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        max_disk_space=max_disk_space,
    )
    # state_dict keys already have block_prefix stripped, e.g. "self_attention.query_key_value.weight"
    if config.model_type == "gpt_oss":
        return block.load_checkpoint_state(state_dict, torch_dtype)
    if isinstance(block, WrappedDeepseekV3Block):
        state_dict = _remap_deepseekv3_expert_state_dict(state_dict)
    elif isinstance(block, WrappedMixtralBlock):
        state_dict = _remap_mixtral_expert_state_dict(state_dict)
    # Bare decoder layers have initialized parameters even before loading.
    # Reject incomplete checkpoints before mutating the block, while allowing
    # optional/generated buffers (e.g. rotary frequencies) to be absent.
    missing_parameters = sorted(set(dict(block.named_parameters())) - state_dict.keys())
    if missing_parameters:
        raise ValueError(
            f"Incomplete checkpoint for {block_prefix} from {model_name}: "
            f"missing required parameters {missing_parameters}"
        )
    result = block.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        logger.warning(f"{block_prefix}: ignoring unexpected checkpoint keys {result.unexpected_keys[:10]}")
    return block.to(dtype=torch_dtype)


StateDict = Dict[str, torch.Tensor]


def _load_state_dict_from_repo(
    model_name: str,
    block_prefix: str,
    *,
    revision: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
    cache_dir: str,
    max_disk_space: Optional[int] = None,
) -> StateDict:
    if always_needs_auth(model_name) and token is None:
        token = True

    index_file = _find_index_file(model_name, revision=revision, token=token, cache_dir=cache_dir)
    if index_file.endswith(".index.json"):  # Sharded model
        path = get_file_from_repo(
            model_name,
            filename=index_file,
            revision=revision,
            use_auth_token=token,
            cache_dir=cache_dir,
        )
        if path is None:
            # _find_index_file() told that a file exists but we can't get it (e.g., it just disappeared)
            raise ValueError(f"Failed to get file {index_file}")

        with open(path) as f:
            index = json.load(f)
        filenames = {
            filename for param_name, filename in index["weight_map"].items() if param_name.startswith(block_prefix)
        }
        if not filenames:
            raise RuntimeError(f"Block {block_prefix}* not found in the index: {index['weight_map']}")
    else:  # Non-sharded model
        filenames = {index_file}
    logger.debug(f"Loading {block_prefix}* from {filenames}")

    state_dict = {}
    for filename in filenames:
        shard_state_dict = _load_state_dict_from_repo_file(
            model_name,
            filename,
            block_prefix=block_prefix,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            max_disk_space=max_disk_space,
        )
        shard_state_dict = {
            param_name[len(block_prefix) :]: param
            for param_name, param in shard_state_dict.items()
            if param_name.startswith(block_prefix)
        }  # Remove unused parameters from memory
        state_dict.update(shard_state_dict)
    
    return state_dict


INDEX_FILES = ["model.safetensors.index.json", "model.safetensors", "pytorch_model.bin.index.json", "pytorch_model.bin"]


def _find_index_file(
    model_name: str, *, revision: Optional[str] = None, token: Optional[Union[str, bool]] = None, cache_dir: str
) -> str:
    # If we have cached weights (e.g., Pickle from older Petals versions), reuse them
    for filename in INDEX_FILES:
        path = get_file_from_repo(
            model_name,
            filename,
            revision=revision,
            use_auth_token=token,
            cache_dir=cache_dir,
            local_files_only=True,
        )
        if path is not None:
            return filename

    # If we don't, prefer Safetensors when possible
    # (we don't download files here since we can't account for max_disk_space in case of large files)
    for filename in INDEX_FILES:
        with suppress(EntryNotFoundError):
            get_hf_file_metadata(hf_hub_url(model_name, filename, revision=revision), token=token)
            return filename

    raise ValueError(
        f"Repo {model_name} does not contain weights in a supported format: files {INDEX_FILES} do not exist"
    )


def _load_state_dict_from_repo_file(
    model_name: str,
    filename: str,
    *,
    block_prefix: Optional[str] = None,
    revision: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
    cache_dir: str,
    max_disk_space: Optional[int] = None,
    delay: float = 30,
) -> StateDict:
    # First, try to find the weights locally
    try:
        with allow_cache_reads(cache_dir):
            path = get_file_from_repo(
                model_name,
                filename,
                revision=revision,
                use_auth_token=token,
                cache_dir=cache_dir,
                local_files_only=True,
            )
            if path is not None:
                return _load_state_dict_from_local_file(path, block_prefix=block_prefix)
    except Exception:
        logger.warning(f"Cache for file {filename} is corrupted, it will be downloaded again", exc_info=True)

    # If not found, ensure that we have enough disk space to download them (maybe remove something)
    while True:
        try:
            with allow_cache_writes(cache_dir):
                url = hf_hub_url(model_name, filename, revision=revision)
                file_size = get_hf_file_metadata(url, token=token).size
                if file_size is not None:
                    free_disk_space_for(file_size, cache_dir=cache_dir, max_disk_space=max_disk_space)
                else:
                    logger.warning(f"Failed to fetch size of file {filename} from repo {model_name}")
                path = get_file_from_repo(
                    model_name,
                    filename,
                    revision=revision,
                    use_auth_token=token,
                    cache_dir=cache_dir,
                    local_files_only=False,
                )
                if path is None:
                    raise RuntimeError(f"File {filename} does not exist in repo {model_name}")
                return _load_state_dict_from_local_file(path, block_prefix=block_prefix)
        except Exception as e:
            logger.warning(f"Failed to load file {filename} from HF Hub (retry in {delay:.0f} sec)", exc_info=True)
            time.sleep(delay)


def _load_state_dict_from_local_file(path: str, *, block_prefix: Optional[str] = None) -> StateDict:
    if path.endswith(".bin"):
        return torch.load(path, map_location="cpu")

    if path.endswith(".safetensors"):
        with safetensors.safe_open(path, framework="pt", device="cpu") as f:
            res = {key: f.get_tensor(key) for key in f.keys() if block_prefix is None or key.startswith(block_prefix)}
            return res

    raise ValueError(f"Unknown weight format: {path}")
