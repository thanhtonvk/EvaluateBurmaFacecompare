import torch
import onnxruntime


def inferenceSession(onnx_path):
    cpu =True
    cuda = not cpu and torch.cuda.is_available()
    # session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    # session_options.add_session_config_entry('session.dynamic_block_base', '4')
    execution_providers = ["CPUExecutionProvider"]
    if cuda:
        cuda_provider_options = {
            "arena_extend_strategy": "kSameAsRequested",
            "cudnn_conv_algo_search": "DEFAULT",
        }
        execution_providers = [
            ("CUDAExecutionProvider", cuda_provider_options),
            "CPUExecutionProvider",
        ]
    print("[utils.onnxruntime] InferenceSession", onnx_path, execution_providers)
    return onnxruntime.InferenceSession(
        onnx_path, providers=execution_providers
    )
