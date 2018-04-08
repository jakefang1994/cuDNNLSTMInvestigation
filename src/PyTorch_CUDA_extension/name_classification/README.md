# Image Classification

### Build
```bash install.bash```
### Test official model
```python official_lstm.py```
### Test custom model
```python custom_lstm.py```

---

```input_size == hidden_size```

Input: ```(seq_length, mini_batch, input_size)```

Weight: ```(num_layer, 2, 4 * hidden_size, input_size) -> flattend 1-D``` // x_weight & h_weight

Bias: ```(num_layer, 2, 4 * hidden_size) -> flattend 1-D```

Weight transpose (```cublasSgeam```): ```(input_size, 4 * hidden_size) -> (4 * hidden_size, input_size)```

Matrix multiplication (```cublasSgemm```): ```(4 * hidden_size, intpu_size) x (input_size, mini_batch) -> (4 * hidden_size, mini_batch)```
