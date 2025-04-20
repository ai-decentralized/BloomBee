"""
The LLaMA model configurations and weight downloading utilities.

Some functions are adopted from https://github.com/alpa-projects/alpa/tree/main/examples/llm_serving/model.
Some configs are adopted from https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/configuration_llama.py
"""

import argparse
import dataclasses
import glob
import os

import numpy as np
from tqdm import tqdm

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


def download_llama_weights_old(model_name, path):
    """Download weights from huggingface."""
    import torch
    from transformers import LlamaForCausalLM

    if "/" in model_name:
        model_name = model_name.split("/")[1].lower()
    path = os.path.join(path, f"{model_name}-np")
    path = os.path.abspath(os.path.expanduser(path))

    if "llama" in model_name:
        hf_model_name = "huggyllama/" + model_name
        model_class = LlamaForCausalLM
    else:
        raise ValueError("Invalid model name: {model_name}")

    print(f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
          f"The downloading and cpu loading can take dozens of minutes. "
          f"If it seems to get stuck, you can monitor the progress by "
          f"checking the memory usage of this process.")

    disable_torch_init()
    model = model_class.from_pretrained(hf_model_name, torch_dtype=torch.float16,
                                        _fast_init=True)
    restore_torch_init()

    os.makedirs(path, exist_ok=True)

    print(f"Convert the weights to numpy format under {path} ...")
    if "llama" in model_name:
        for name, param in tqdm(list(model.model.named_parameters())):
            param_path = os.path.join(path, name)
            with open(param_path, "wb") as f:
                np.save(f, param.cpu().detach().numpy())
    else:
        raise ValueError("Invalid model name: {model_name}")


global torch_linear_init_backup
global torch_layer_norm_init_backup


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    global torch_linear_init_backup
    global torch_layer_norm_init_backup

    torch_linear_init_backup = torch.nn.Linear.reset_parameters
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)

    torch_layer_norm_init_backup = torch.nn.LayerNorm.reset_parameters
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def restore_torch_init():
    """Rollback the change made by disable_torch_init."""
    import torch
    setattr(torch.nn.Linear, "reset_parameters", torch_linear_init_backup)
    setattr(torch.nn.LayerNorm, "reset_parameters", torch_layer_norm_init_backup)


def disable_hf_llama_init():
    """
    Disable the redundant default initialization to accelerate model creation.
    """
    import transformers

    setattr(transformers.models.llama.modeling_llama.LlamaPreTrainedModel,
            "_init_weights", lambda *args, **kwargs: None)


def download_llama_weights(model_name, path):
    from huggingface_hub import snapshot_download
    import torch

    print(f"Load the pre-trained pytorch weights of {model_name} from huggingface. "
          f"The downloading and cpu loading can take dozens of minutes. "
          f"If it seems to get stuck, you can monitor the progress by "
          f"checking the memory usage of this process.")
    if "llama" in model_name:
        hf_model_name = "huggyllama/" + model_name

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="llama-7b")
    parser.add_argument("--path", type=str, default="/tmp/data/llama_weights")
    args = parser.parse_args()

    download_llama_weights(args.model, args.path)