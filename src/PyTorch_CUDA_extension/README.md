# PyTorch_CUDA_extension

### Build
```bash install.bash```
### Test
```python test.py```

---

```input_size == hidden_size```

Input: ```(seq_length, mini_batch, input_size)``` // input_size must be the lowest dim, effects matrix multiplication

Weight: ```(num_layer, 2, 4 * hidden_size, input_size) -> flattend 1-D``` // x_weight & h_weight

Bias: ```(num_layer, 2, 4 * hidden_size) -> flattend 1-D```

Weight transpose (```cublasSgeam```): ```(4 * hidden_size, input_size) -> (input_size, 4 * hidden_size)```

Matrix multiplication (```cublasSgemm```): ```(mini_batch, input_size) x (intpu_size, 4 * hidden_size) -> (mini_batch, 4 * hidden_size)```
