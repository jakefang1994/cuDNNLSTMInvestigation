# PyTorch_CUDA_extension

#### Change from official PyTorch model to custom model in Python

- Replace model (torch.nn.Module) and corresponding calling API
- Manually load pre-trained weight and bias of LSTM layer by torch.load(ckpt), and flatten them to 1-D arrays
- Directly load pre-trained weight and bias of fully-connected layer in custom model
- Flatten input data to 1-D array
