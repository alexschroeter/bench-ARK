# Cellpose

## ToDo

### Training

- Add Ground truth comparison
- ✅ The downloaded data needs to end up in the data folder of the benchmark, similar to what happens for the artificial benchmark.
- ✅ The training benchmark needs to store the resulting model as part of the result. This also needs to be checked if it is identical in all iterations.
- ✅ The training benchmark needs to store the mask as part of the result and it needs to be checked in each iteration.
- From the mask we need to create the Plots, which shows the differences between different GPUs.
- From the mask we need to create the Plot, that shows the difference to the Ground Truth which we trained on.
- ✅ From the model we need to show how different the results are which we can reference by the resulting mask.

### Aritificial Inference Benchmark

- Check if each iteration of the benchmark produces the identical result (mask).

## Completed Tasks

#### Device Context Bug Fix ✅
- Fixed device context handling in training benchmark evaluation phase
- Models now properly use the device they were trained on for evaluation
- See: `BUG_FIX_DEVICE_CONTEXT.md`

#### Model Comparison Features ✅
- Implemented comprehensive model comparison visualizations:
  - Similarity matrices (L2, cosine similarity, max diff)
  - Statistics tables (per-model metrics)
  - Weight distribution comparisons
- See: `MODEL_COMPARISON_FEATURES.md`

#### Training Analysis ✅
- Discovered Cellpose learning rate warmup issue (n_epochs=1 means LR=0, no training)
- Analyzed 10-epoch training results showing actual parameter differences
- Documented cross-device/flavour model variations
- See: `BUG_DISCOVERY_NO_TRAINING.md`, `10_EPOCH_TRAINING_ANALYSIS.md`

#### File Size Investigation ✅
- Explained file size differences between CPU/GPU models
- Identified device metadata as cause (4,928 byte difference)
- Parameter values confirmed identical despite file size difference
- See: `FILE_SIZE_INVESTIGATION.md`

#### Deterministic Training Support ✅
- Researched Cellpose codebase - NO built-in deterministic support found
- Added `deterministic` and `seed` configuration parameters to training benchmark
- Implemented comprehensive deterministic settings:
  - Python, NumPy, PyTorch random seeds
  - cuDNN deterministic mode
  - PyTorch deterministic algorithms (when available)
- See: `DETERMINISTIC_TRAINING.md`

### Next Steps

#### Testing Deterministic Training
1. Run training with `deterministic: false` (baseline)
2. Run training with `deterministic: true, seed: 42` (reproducibility test)
3. Compare models from same device (should be identical with deterministic=true)
4. Compare models across devices (may still differ due to hardware)
5. Test different seeds (should produce different but reproducible models)


## Reproducibility

```bash 
/usr/local/lib/python3.12/dist-packages/torch/nn/modules/conv.py:543: UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:315.)
  return F.conv2d(
/usr/local/lib/python3.12/dist-packages/segment_anything/modeling/image_encoder.py:231: UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:315.)
  attn = (q * self.scale) @ k.transpose(-2, -1)
/usr/local/lib/python3.12/dist-packages/torch/functional.py:373: UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:315.)
  return _VF.einsum(equation, operands)  # type: ignore[attr-defined]
/usr/local/lib/python3.12/dist-packages/segment_anything/modeling/image_encoder.py:237: UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:315.)
  x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
/usr/local/lib/python3.12/dist-packages/cellpose/vit_sam.py:76: UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:315.)
  x1 = F.conv_transpose2d(x1, self.W2, stride = self.ps, padding = 0)
/usr/local/lib/python3.12/dist-packages/torch/autograd/graph.py:841: UserWarning: Deterministic behavior was enabled with either `torch.use_deterministic_algorithms(True)` or `at::Context::setDeterministicAlgorithms(true)`, but this operation is not deterministic because it uses CuBLAS and you have CUDA >= 10.2. To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application: CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8. For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:315.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
/usr/local/lib/python3.12/dist-packages/torch/autograd/graph.py:841: UserWarning: upsample_linear1d_backward_out_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:157.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
```

- Why is dataset determinsitic CPU result the same?

- ✅ Add setting the environment variable for deterministic execution. `CUBLAS_WORKSPACE_CONFIG=:4096:8`
- ✅ Add the ground truth comparison to the 500 epochs training (with and without deterministic)

- This is the reason why CUDA does not give predictable results even on the same hardware.
```bash
/usr/local/lib/python3.12/dist-packages/torch/autograd/graph.py:841: UserWarning: upsample_linear1d_backward_out_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (Triggered internally at /pytorch/aten/src/ATen/Context.cpp:157.)
```
