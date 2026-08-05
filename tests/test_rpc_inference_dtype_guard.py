import torch
from hivemind.proto import runtime_pb2
from hivemind.utils.tensor_descr import BatchTensorDescriptor

from bloombee.client.inference_session import (
    _prepare_rpc_inference_tensor_for_wire,
    _server_hidden_states_wire_dtype,
)


def test_hidden_states_wire_dtype_uses_server_schema_dtype():
    server_schema = (
        BatchTensorDescriptor(1, dtype=torch.float16, compression=runtime_pb2.CompressionType.NONE),
    )
    server_dtype = _server_hidden_states_wire_dtype(server_schema)
    tensor = torch.randn(2, 3, 4, dtype=torch.float32)

    wire_tensor, proto, debug = _prepare_rpc_inference_tensor_for_wire(
        tensor,
        "hidden_states",
        runtime_pb2.CompressionType.NONE,
        server_dtype,
    )

    assert wire_tensor.dtype == torch.float16
    assert proto.dtype == torch.float16
    assert debug["compute_dtype"] == "float32"
    assert debug["schema_dtype"] == "float16"
    assert debug["wire_dtype"] == "float16"
    assert debug["dtype_guard_applied"] == 1
    assert debug["upcast_suspect"] == 1


def test_non_hidden_states_keep_live_dtype():
    server_dtype = torch.float16
    keep_indices = torch.arange(4, dtype=torch.int64)

    wire_tensor, proto, debug = _prepare_rpc_inference_tensor_for_wire(
        keep_indices,
        "keep_indices",
        runtime_pb2.CompressionType.NONE,
        server_dtype,
    )

    assert wire_tensor.dtype == torch.int64
    assert proto.dtype == torch.int64
    assert debug["compute_dtype"] == "int64"
    assert debug["schema_dtype"] == ""
    assert debug["wire_dtype"] == "int64"
    assert debug["dtype_guard_applied"] == 0
    assert debug["upcast_suspect"] == 0
