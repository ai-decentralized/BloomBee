"""
Utils for fetching pretrained model parts. Currently, this relies on huggingface transformers' from_pretrained code.
If necessary, one can rewrite this to implement a different behavior, such as:
 - loading files from a local data source (e.g. S3)
 - load files via BitTorrent ( https://pypi.org/project/libtorrent/ ) or IPFS( https://docs.ipfs.io/how-to )
 - fetch the weights over IPoAC, using a fleet of trained pigeons ( http://www.faqs.org/rfcs/rfc1149.html )

"""
import json
import time
from contextlib import suppress
from typing import Dict, Optional, Union

import safetensors
import torch
import torch.nn as nn
from accelerate import init_empty_weights
# from accelerate.utils import set_module_tensor_to_device
from hivemind.utils.logging import get_logger
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from huggingface_hub.utils import EntryNotFoundError
from transformers import PretrainedConfig, PreTrainedModel
from transformers.utils import get_file_from_repo

from bloombee.constants import DTYPE_MAP
from bloombee.models.mixtral import WrappedMixtralBlock
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
import numpy as np
# import pdb

from bloombee.models.llama.flex_llama import load_weights_from_pytorch_model





DUMMY_WEIGHT = "_DUMMY_"  # Use dummy weights for benchmark purposes
logger = get_logger(__name__)


def load_pretrained_block(
    model_name: str,
    block_index: int,
    env: ExecutionEnv,
    policy: Policy,
    weight_home: array_1d,
    path: str,
    *,
    config: Optional[PretrainedConfig] = None,
    torch_dtype: Union[torch.dtype, str] = "auto",
    revision: Optional[str] = None,
    token: Optional[Union[str, bool]] = None,
    cache_dir: Optional[str] = None,
    max_disk_space: Optional[int] = None,
) -> nn.Module:
    if config is None:
        config = AutoDistributedConfig.from_pretrained(model_name, use_auth_token=token)
    if cache_dir is None:
        cache_dir = DEFAULT_CACHE_DIR
    
    assert torch_dtype in DTYPE_MAP.values(), f"torch_dtype must be one of {list(DTYPE_MAP.values())}"
    torch_dtype = resolve_block_dtype(config, torch_dtype)
    
    with init_empty_weights():
        print('load_pretrained_block : init_empty_weights() ') 
        block = get_model_block(config, env, policy, weight_home, path, layer_idx=block_index)
    
    block_prefix = f"{config.block_prefix}.{block_index}."
    state_dict = _load_state_dict_from_repo(
        model_name,
        block_prefix,
        revision=revision,
        token=token,
        cache_dir=cache_dir,
        max_disk_space=max_disk_space,
    )
    
    # 将权重加载到模型中
    for param_name, param in state_dict.items():
        if not str(param.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
            param = param.to(torch_dtype)
        set_module_tensor_to_device(block, param_name, "cpu", value=param, dtype=param.dtype)
    
    # # 使用 FlexGen 的权重加载方式
    # try:
    #     load_weights_from_pytorch_model(block, policy, env, weight_home, block_index)
    #     logger.info(f"Loaded {model_name} block {block_index} with FlexGen weight management")
    # except Exception as e:
    #     logger.warning(f"Failed to load weights with FlexGen: {e}")
    #     logger.info(f"Loaded {model_name} block {block_index} with direct parameter assignment")
    
    return block


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
    # print('index_file :', index_file)
    # print('cache_dir : ', cache_dir)
    if index_file.endswith(".index.json"):  # Sharded model
        path = get_file_from_repo(model_name, filename=index_file, use_auth_token=token, cache_dir=cache_dir)
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
        # print('filename ', filename)
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
    # print('from_pretrained.py  _load_state_dict_from_repo_file(): filename ', filename)
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
            # print('path ', path)
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
                # print('from_pretrained.py  _load_state_dict_from_repo_file(): filename ', filename)
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
            # print('res ', res)
            return res

    raise ValueError(f"Unknown weight format: {path}")


    #state_dict={
        # 'model.layers.0.input_layernorm.weight': tensor([0.0446, 0.0191, 0.0188,  ..., 0.0319, 0.0243, 0.0264], dtype=torch.float16), 
        # 'model.layers.0.mlp.down_proj.weight': tensor([[-0.0075, -0.0083,  0.0520,  ..., -0.0033,  0.0243,  0.0004],
        # ...
        # 'model.layers.0.self_attn.v_proj.weight': tensor([[ 0.0061,  0.0045,  0.0020,  ..., -0.0079,  0.0114,  0.0150],
        #  ...
        # [ 0.0099, -0.0020, -0.0023,  ..., -0.0068,  0.0156,  0.0027]], dtype=torch.float16)
        # }


# def init_weight_list(state_dict, policy, env, block):
#     dev_percents = [policy.w_disk_percent, policy.w_cpu_percent, policy.w_gpu_percent]
#     dev_choices = [env.disk, env.cpu, env.gpu]
#     print('block')
#     # import pdb; pdb.set_trace()
#     sizes=[]
#     for param_name, _ in block.named_parameters():
#         assert param_name in state_dict, f"{param_name} not in state dict"
#         param = state_dict[param_name]
#         cur_shape = np.array(param.size())
#         # print('current shape ', cur_shape)
#         cur_size = np.prod(cur_shape)
#         sizes.append(cur_size)
#         # print('current size ', cur_size)
#     # sizes=[16777216, 16777216, 16777216, 16777216, 45088768, 45088768, 45088768, 4096, 4096]  
#     sizes_cumsum = np.cumsum(sizes)   
#     # sizes_cumsum  [ 16777216  33554432  50331648  67108864 112197632 157286400 202375168 202379264 202383360]
#     # print('sizes_cumsum ', sizes_cumsum)
    
#     ret = []
#     i=0
#     for param_name, _param in block.named_parameters():
#         mid_percent = (sizes_cumsum[i] - sizes[i] / 2) / sizes_cumsum[-1]
#         home = get_choice(mid_percent * 100, dev_percents, dev_choices)
#         # print('home ', home)
#         # print('state_dict', state_dict)
#         # print('state_dict[param_name]', state_dict[param_name])
#         # print('param_name', param_name)
#         param = state_dict[param_name]
#         # print()
#         shape = param.size()
#         dtype = param.dtype
        
#         if len(shape) < 2:
#             pin_memory = True
#             compress = False
#         else:
#             pin_memory = policy.pin_weight
#             compress = policy.compress_weight

#         if not compress:
#             weight = home.allocate(shape, dtype, pin_memory=pin_memory)
#             weight.load_from_state_dict(param) ###############
#             print('weight.shape ', weight.shape)
#             if DUMMY_WEIGHT not in filename:
#                 weight.load_from_np_file(weight_specs[i][2])
#             else:
#                 weight.load_from_np(np.ones(shape, dtype))
#                 #weight.load_from_np(np.random.rand(*shape).astype(dtype))
#         else: # compress
#             weight = home.compressed_device.allocate(
#                 shape, dtype, policy.comp_weight_config, pin_memory=pin_memory)

#             if DUMMY_WEIGHT not in filename:
#                 # weight.load_from_np_file(weight_specs[i][2])
#                 weight.load_from_state_dict(param)
#             else:
#                 for i in range(2):
#                     x = weight.data[i]
#                     x.load_from_np(np.ones(x.shape, torch_dtype_to_np_dtype[x.dtype]))
#         i+=1
#         ret.append(weight)
#         # set_module_tensor_to_device(block, param_name, weight.device.dev, weight, value=param, dtype=param.dtype)
#         # set_module_tensor_to_device(block, param_name, "cpu", weight, value=param, dtype=param.dtype)
        
#         # block._parameters[tmp_name]= weight.data
        
     
       

    

def get_choice(cur_percent, percents, choices):
    percents = np.cumsum(percents)
    assert np.abs(percents[-1] - 100) < 1e-5

    for i in range(len(percents)):
        if cur_percent < percents[i]:
            return choices[i]
    return choices[-1]

def set_module_tensor_to_device(
    module: nn.Module,
    tensor_name: str,
    device: Union[int, str, torch.device],
    # weight: TorchTensor,
    value: Optional[torch.Tensor] = None,
    dtype: Optional[Union[str, torch.dtype]] = None,
    fp16_statistics: Optional[torch.HalfTensor] = None,
    tied_params_map: Optional[Dict[int, Dict[torch.device, torch.Tensor]]] = None,
):
    
    # Recurse if needed
    if "." in tensor_name:
        splits = tensor_name.split(".")
        for split in splits[:-1]:
            new_module = getattr(module, split)
            if new_module is None:
                raise ValueError(f"{module} has no attribute {split}.")
            module = new_module
        tensor_name = splits[-1]
    # import pdb; pdb.set_trace() 
    # module._parameters: OrderedDict([('weight', Parameter containing: tensor(..., device='meta', size=(4096, 4096), requires_grad=True)), ('bias', None)])
    if tensor_name not in module._parameters and tensor_name not in module._buffers:
        raise ValueError(f"{module} does not have a parameter or a buffer named {tensor_name}.")
    is_buffer = tensor_name in module._buffers #-------
    old_value = getattr(module, tensor_name)#---------

    if (
        value is not None
        and tied_params_map is not None
        and value.data_ptr() in tied_params_map
        and device in tied_params_map[value.data_ptr()]
    ):
        module._parameters[tensor_name] = tied_params_map[value.data_ptr()][device]
        return
    elif (
        tied_params_map is not None
        and old_value.data_ptr() in tied_params_map
        and device in tied_params_map[old_value.data_ptr()]
    ):
        module._parameters[tensor_name] = tied_params_map[old_value.data_ptr()][device]
        return

    if old_value.device == torch.device("meta") and device not in ["meta", torch.device("meta")] and value is None:
        raise ValueError(f"{tensor_name} is on the meta device, we need a `value` to put in on {device}.")
    param = module._parameters[tensor_name] if tensor_name in module._parameters else None
    param_cls = type(param) # param_cls is <class 'torch.nn.parameter.Parameter'>
    if value is not None:
        # We can expect mismatches when using bnb 4bit since Params4bit will reshape and pack the weights.
        # In other cases, we want to make sure we're not loading checkpoints that do not match the config.
        if old_value.shape != value.shape and param_cls.__name__ != "Params4bit":
            raise ValueError(
                f'Trying to set a tensor of shape {value.shape} in "{tensor_name}" (which has shape {old_value.shape}), this looks incorrect.'
            )

        if dtype is None:
            # For compatibility with PyTorch load_state_dict which converts state dict dtype to existing dtype in model
            value = value.to(old_value.dtype)
        elif not str(value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
            value = value.to(dtype) #------------

    device_quantization = None #------------
    
    with torch.no_grad(): #------------
        # leave it on cpu first before moving them to cuda
        # # fix the case where the device is meta, we don't want to put it on cpu because there is no data =0
        if (
            param is not None
            and param.device.type != "cuda"
            and torch.device(device).type == "cuda"
            and param_cls.__name__ in ["Int8Params", "FP4Params", "Params4bit"]
        ):
            device_quantization = device
            device = "cpu"
        # # `torch.Tensor.to(<int num>)` is not supported by `torch_npu` (see this [issue](https://github.com/Ascend/pytorch/issues/16)).
        if isinstance(device, int):
            if is_npu_available():
                device = f"npu:{device}"
            elif is_mlu_available():
                device = f"mlu:{device}"
            elif is_musa_available():
                device = f"musa:{device}"
            elif is_xpu_available():
                device = f"xpu:{device}"
        if "xpu" in str(device) and not is_xpu_available():
            raise ValueError(f'{device} is not available, you should use device="cpu" instead')
        if value is None:
            new_value = old_value.to(device)
            if dtype is not None and device in ["meta", torch.device("meta")]:
                if not str(old_value.dtype).startswith(("torch.uint", "torch.int", "torch.bool")):
                    new_value = new_value.to(dtype)

                if not is_buffer:
                    module._parameters[tensor_name] = param_cls(new_value, requires_grad=old_value.requires_grad)
        elif isinstance(value, torch.Tensor):
            new_value = value.to(device) #------------
        else:
            new_value = torch.tensor(value, device=device)
        if device_quantization is not None:
            device = device_quantization
        if is_buffer:
            module._buffers[tensor_name] = new_value
        elif value is not None or not check_device_same(torch.device(device), module._parameters[tensor_name].device):
            param_cls = type(module._parameters[tensor_name]) #------------
            kwargs = module._parameters[tensor_name].__dict__
            if param_cls.__name__ in ["Int8Params", "FP4Params", "Params4bit"]:
                if param_cls.__name__ == "Int8Params" and new_value.dtype == torch.float32:
                    # downcast to fp16 if any - needed for 8bit serialization
                    new_value = new_value.to(torch.float16)
                # quantize module that are going to stay on the cpu so that we offload quantized weights
                if device == "cpu" and param_cls.__name__ == "Int8Params":
                    new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to(0).to("cpu")
                    new_value.CB = new_value.CB.to("cpu")
                    new_value.SCB = new_value.SCB.to("cpu")
                else:
                    new_value = param_cls(new_value, requires_grad=old_value.requires_grad, **kwargs).to(device)
            elif param_cls.__name__ in ["QTensor", "QBitsTensor"]:
                new_value = torch.nn.Parameter(new_value, requires_grad=old_value.requires_grad).to(device)
            elif param_cls.__name__ in ["AffineQuantizedTensor"]:
                new_value = torch.nn.Parameter(
                    param_cls(
                        new_value.layout_tensor,
                        new_value.block_size,
                        new_value.shape,
                        new_value.quant_min,
                        new_value.quant_max,
                        new_value.zero_point_domain,
                    ),
                    requires_grad=old_value.requires_grad,
                ).to(device)
            else:
                new_value = param_cls(new_value, requires_grad=old_value.requires_grad).to(device)

            # module._parameters[tensor_name] = weight.data #######------------
            # compare_tensors(new_value, weight.data)
            
            module._parameters[tensor_name] = new_value #######------------
            if fp16_statistics is not None:
                module._parameters[tensor_name].SCB = fp16_statistics.to(device)
                del fp16_statistics
            # as we put the weight to meta, it doesn't have SCB attr anymore. make sure that it is not a meta weight
            if (
                module.__class__.__name__ == "Linear8bitLt"
                and getattr(module.weight, "SCB", None) is None
                and str(module.weight.device) != "meta"
            ):
                # quantize only if necessary
                device_index = torch.device(device).index if torch.device(device).type == "cuda" else None
                if not getattr(module.weight, "SCB", None) and device_index is not None:
                    if module.bias is not None and module.bias.device.type != "meta":
                        # if a bias exists, we need to wait until the bias is set on the correct device
                        module = module.cuda(device_index)
                    elif module.bias is None:
                        # if no bias exists, we can quantize right away
                        module = module.cuda(device_index)
            elif (
                module.__class__.__name__ == "Linear4bit"
                and getattr(module.weight, "quant_state", None) is None
                and str(module.weight.device) != "meta"
            ):
                # quantize only if necessary
                device_index = torch.device(device).index if torch.device(device).type == "cuda" else None
                if not getattr(module.weight, "quant_state", None) and device_index is not None:
                    module.weight = module.weight.cuda(device_index)
    if device != "cpu": #---------
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    

def compare_tensors(new_value, weight_data):  
    print('new_value ', new_value.device)
    print('weight_data ', weight_data.device)
    print()
    # Compare data types  
    if new_value.dtype != weight_data.dtype:  
        return f"Data types are different: {new_value.dtype} vs {weight_data.dtype}"  
    
    # Compare values  
    if torch.equal(new_value.to('cpu'), weight_data.to('cpu')):  
        return "The tensors are equal in value."  
    else:  
        return "The tensors are not equal in value."  
    