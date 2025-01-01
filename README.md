# PKU_machine_learning_term_proj

Author: Xu Yue, Yi Xiaoyu, An Ruibo, Zhang Lingxin

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

**The implementation of function above has been checked to be correct.**

In `ilc.py` is the implementation of iterative lattice construction algorithm from the original paper. Directly modified from the pseudocode. 

In `test_2d.py` is the test code for a 2d-case visualization. It can output a hexagonal lattice as expected. 
