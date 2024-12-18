# PKU_machine_learning_term_proj

Author: Xu Yue, Yi Xiaoyu

## Quickstart

```bash
pip install -r requirements.txt
python test_2d.py
```

## Source codes

In `utils.py` are the implementations of utility functions including the following:

- `GRAN`
- `URAN`
- `CLP`
- `RED`
- `ORTH`
  
Please refer to paper "Optimization and Identification of Lattice Quantizers" for the detailed definition of each function.

Currently Xiaoyu has made sure that the implementations of function `GRAN`, `URAN`, `RED` and `ORTH` are consistent with what being defined in the original paper. Yet a double-check is recommended.

**The implementation of function `CLP` can not be made sure to be correct. Please refer to the `TODO` in `utils.py` for more information.**

In `ilc.py` is the implementation of iterative lattice construction algorithm from the original paper. Directly modified from the pseudocode.

In `test_2d.py` is the test code for a 2d-case visualization. It should output a hexagonal lattice as expected however it does not. **Finish the implementation of `CLP` first and investigate the reason for the unexpected result after making sure all utility functions especially `CLP` are correctly implemented.**
