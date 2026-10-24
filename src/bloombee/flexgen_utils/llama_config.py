"""
The LLaMA model configurations and weight downloading utilities.

Some functions are adopted from https://github.com/alpa-projects/alpa/tree/main/examples/llm_serving/model.
Some configs are adopted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/configuration_llama.py
"""

import argparse
import dataclasses
import glob
import hashlib
import json
import os
from typing import Optional

import numpy as np
from tqdm import tqdm

from bloombee.utils.debug import dprint

FLEXGEN_NP_FORMAT_VERSION = "np-v1"
FLEXGEN_NP_MANIFEST_NAME = ".bloombee_np_manifest.json"
FLEXGEN_NP_COMPLETE_NAME = ".bloombee_np_converted"


def flexgen_np_cache_identity(source: str, revision: Optional[str] = None) -> str:
    """Stable identity for a FlexGen numpy conversion cache.

    Basename-only cache directories collide across orgs, local paths, and
    revisions. Hash the resolved source, revision, and conversion format.
    """
    if source and os.path.isdir(source):
        source_key = os.path.abspath(os.path.expanduser(source))
    else:
        source_key = str(source or "")
    payload = f"{source_key}|{revision or ''}|{FLEXGEN_NP_FORMAT_VERSION}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def flexgen_np_cache_dirname(source: str, model_name: str, revision: Optional[str] = None) -> str:
    safe_name = str(model_name).replace("/", "_").replace(" ", "_") or "llama"
    return f"{safe_name}-{flexgen_np_cache_identity(source, revision)}-np"


def flexgen_np_cache_dir(path: str, source: str, model_name: str, revision: Optional[str] = None) -> str:
    return os.path.abspath(os.path.expanduser(os.path.join(path, flexgen_np_cache_dirname(source, model_name, revision))))


def _write_flexgen_np_manifest(out_dir: str, *, source: str, revision: Optional[str], files: list) -> None:
    manifest = {
        "source": source,
        "revision": revision,
        "format_version": FLEXGEN_NP_FORMAT_VERSION,
        "files": sorted(files),
    }
    with open(os.path.join(out_dir, FLEXGEN_NP_MANIFEST_NAME), "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    with open(os.path.join(out_dir, FLEXGEN_NP_COMPLETE_NAME), "w") as handle:
        handle.write("ok\n")


def _flexgen_np_cache_ready(out_dir: str, *, source: str, revision: Optional[str]) -> bool:
    complete = os.path.join(out_dir, FLEXGEN_NP_COMPLETE_NAME)
    embed = os.path.join(out_dir, "embed_tokens.weight")
    manifest_path = os.path.join(out_dir, FLEXGEN_NP_MANIFEST_NAME)
    if not (os.path.isfile(complete) and os.path.isfile(embed)):
        return False
    if not os.path.isfile(manifest_path):
        return False
    try:
        with open(manifest_path) as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    if manifest.get("format_version") != FLEXGEN_NP_FORMAT_VERSION:
        return False
    if str(manifest.get("source") or "") != str(source or ""):
        return False
    if str(manifest.get("revision") or "") != str(revision or ""):
        return False
    return True

@dataclasses.dataclass(frozen=True)
class LlamaConfig:
    name: str="llama-7b"
    vocab_size: int=32000
    type_vocab_size=2
    input_dim: int=4096
    intermediate_size: int=11008
    num_hidden_layers: int=32
    n_head: int=32
    hidden_act: str="silu"
    max_position_embeddings: int=2048
    initializer_range: float=0.02
    rms_norm_eps: float=1e-6
    pad_token_id: int=0
    tie_word_embeddings: bool=False
    dtype: type=np.float16

    def model_bytes(self):
        V = self.vocab_size
        H = self.input_dim
        L = self.num_hidden_layers
        I = self.intermediate_size
        num_params = L*(4*H*H + 3*I*H + 2*H) + V*H*2 + H
        return num_params * 2    

    def cache_bytes(self, batch_size, seq_len):
        return 2 * batch_size * seq_len * self.num_hidden_layers * self.input_dim * 2

    def hidden_bytes(self, batch_size, seq_len):
        return batch_size * seq_len * self.input_dim * 2

def get_llama_config(name, **kwargs):
    if "/" in name:
        name = name.split("/")[1]
    name = name.lower()

    arch_name = name

    if arch_name == "llama-7b":
        config = LlamaConfig(name=name, input_dim=4096, n_head=32, num_hidden_layers=32, intermediate_size=11008)
    elif arch_name == "llama-13b":
        config = LlamaConfig(name=name, input_dim=5120, n_head=40, num_hidden_layers=40, intermediate_size=13824)
    elif arch_name == "llama-30b":
        config = LlamaConfig(name=name, input_dim=6656, n_head=52, num_hidden_layers=60, intermediate_size=17920)
    elif arch_name == "llama-65b":
        config = LlamaConfig(name=name, input_dim=8192, n_head=64, num_hidden_layers=80, intermediate_size=22016)
    else:
        raise ValueError(f"Invalid model name: {name}")
    
    return dataclasses.replace(config, **kwargs)


def convert_local_llama_weights(src_model_dir, model_name, path, revision: Optional[str] = None):
    """Convert local HF llama weights (safetensors or .bin) to FlexGen
    numpy-layout cache at ``{path}/{model_name}-{identity}-np/``.

    FlexGen reads weights as one ``.weight`` file per parameter via
    ``np.load``. Target layout:
        {path}/{model_name}-{identity}-np/layers.{i}.self_attn.q_proj.weight
        {path}/{model_name}-{identity}-np/embed_tokens.weight
        ...

    This is the non-downloading analog of ``download_llama_weights``,
    letting the server bootstrap from a model the user has already
    downloaded without making HF calls.
    """
    import torch

    source_key = os.path.abspath(os.path.expanduser(src_model_dir)) if os.path.isdir(src_model_dir) else str(src_model_dir)
    out_dir = flexgen_np_cache_dir(path, source_key, model_name, revision)
    if _flexgen_np_cache_ready(out_dir, source=source_key, revision=revision):
        dprint(f"FlexGen numpy cache already present at {out_dir}; skipping conversion.")
        return out_dir
    os.makedirs(out_dir, exist_ok=True)

    dprint(f"Converting {src_model_dir} → FlexGen numpy layout at {out_dir}")

    # Load state dict: prefer safetensors, fall back to .bin.
    state: dict = {}
    safetensor_files = sorted(glob.glob(os.path.join(src_model_dir, "*.safetensors")))
    if safetensor_files:
        from safetensors import safe_open
        for sf in tqdm(safetensor_files, desc="Read safetensors"):
            with safe_open(sf, framework="pt", device="cpu") as f:
                for k in f.keys():
                    state[k] = f.get_tensor(k)
    else:
        bin_files = sorted(glob.glob(os.path.join(src_model_dir, "*.bin")))
        if not bin_files:
            raise FileNotFoundError(
                f"No *.safetensors or *.bin found in {src_model_dir}"
            )
        for bf in tqdm(bin_files, desc="Read .bin"):
            state.update(torch.load(bf, map_location="cpu"))

    written = []
    for name, param in tqdm(list(state.items()), desc="Convert → np"):
        stripped = name.replace("model.", "")
        stripped = stripped.replace("final_layer_norm", "layer_norm")
        param_path = os.path.join(out_dir, stripped)
        os.makedirs(os.path.dirname(param_path), exist_ok=True)
        arr = param.detach().to(torch.float16).cpu().numpy()
        with open(param_path, "wb") as f:
            np.save(f, arr)
        written.append(stripped)

    if "embed_tokens.weight" not in written and not os.path.isfile(os.path.join(out_dir, "embed_tokens.weight")):
        raise FileNotFoundError(
            f"FlexGen conversion of {src_model_dir} did not produce embed_tokens.weight"
        )
    tmp_manifest = os.path.join(out_dir, FLEXGEN_NP_MANIFEST_NAME + ".tmp")
    _write_flexgen_np_manifest(out_dir, source=source_key, revision=revision, files=written)
    if os.path.isfile(tmp_manifest):
        os.replace(tmp_manifest, os.path.join(out_dir, FLEXGEN_NP_MANIFEST_NAME))
    return out_dir


def download_llama_weights(model_name, path):
    from huggingface_hub import snapshot_download
    import torch

    dprint(f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
          f"The downloading and cpu loading can take dozens of minutes. "
          f"If it seems to get stuck, you can monitor the progress by "
          f"checking the memory usage of this process.")
    if "llama" in model_name.lower():
        # Remove -hf suffix if present, as huggyllama repos don't use it
        clean_name = model_name.replace("-hf", "")
        hf_model_name = "huggyllama/" + clean_name
    else:
        raise ValueError(f"Invalid model name: {model_name}. Only llama models are supported.")

    folder = snapshot_download(hf_model_name, allow_patterns="*.bin")
    bin_files = glob.glob(os.path.join(folder, "*.bin"))

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))
    os.makedirs(path, exist_ok=True)

    for bin_file in tqdm(bin_files, desc="Convert format"):
        state = torch.load(bin_file)
        for name, param in tqdm(state.items(), leave=False):
            name = name.replace("model.", "")
            name = name.replace("final_layer_norm", "layer_norm")
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())


def _looks_like_hf_repo_id(value: Optional[str]) -> bool:
    """True for HuggingFace ids like `huggyllama/llama-7b`, not filesystem paths.

    Older LLaMA configs bake a converter machine path into `_name_or_path`
    (e.g. `/home/sgugger/tmp/llama/llama-7b/`). Those strings contain `/` but
    are not Hub repo ids and must not be passed to `snapshot_download`.
    """
    if not value or not isinstance(value, str):
        return False
    value = value.strip()
    if not value or os.path.isabs(value) or os.path.exists(value):
        return False
    if value.startswith(".") or "\\" in value or ":" in value:
        return False
    parts = [part for part in value.split("/") if part]
    if len(parts) not in (1, 2):
        return False
    return all(part not in (".", "..") for part in parts)


def resolve_flexgen_llama_weights(
    *,
    path: str,
    model_name: str,
    raw_path: Optional[str] = None,
    local_src_dir: Optional[str] = None,
    revision: Optional[str] = None,
    token=None,
) -> str:
    """Return a complete FlexGen numpy cache directory for this source/revision."""
    dummy = "_DUMMY_"
    if dummy in (model_name or "") or dummy in (raw_path or "") or dummy in (path or ""):
        return os.path.abspath(os.path.expanduser(os.path.join(path, f"{model_name}-np")))

    if local_src_dir is not None:
        return convert_local_llama_weights(local_src_dir, model_name, path, revision=revision)

    if raw_path and os.path.isdir(raw_path):
        return convert_local_llama_weights(raw_path, model_name, path, revision=revision)

    if _looks_like_hf_repo_id(raw_path):
        from huggingface_hub import snapshot_download

        src_dir = snapshot_download(
            raw_path,
            revision=revision,
            token=token,
            allow_patterns=["*.safetensors", "*.bin", "*.json"],
        )
        return convert_local_llama_weights(src_dir, model_name, path, revision=revision)

    if str(model_name).startswith("llama-"):
        from huggingface_hub import snapshot_download

        clean_name = str(model_name).replace("-hf", "")
        src_dir = snapshot_download(
            "huggyllama/" + clean_name,
            revision=revision,
            token=token,
            allow_patterns=["*.safetensors", "*.bin", "*.json"],
        )
        return convert_local_llama_weights(src_dir, model_name, path, revision=revision)

    raise FileNotFoundError(
        f"Cannot resolve FlexGen weights for model_name={model_name!r} raw_path={raw_path!r}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama-7b")
    parser.add_argument("--path", type=str, default="/tmp/data/llama_weights")
    args = parser.parse_args()

    download_llama_weights(args.model, args.path)
