# PKU_machine_learning_term_proj

Author: Xu Yue, Yi Xiaoyu, An Ruibo, Zhang Lingxin

## Quickstart

```bash
pip install -r requirements.txt
python test_2d.py
```

## Source codes

### Algorithm implementation

In `utils.py` are the implementations of utility functions including the following:

- `GRAN`
- `URAN`
- `CLP`
- `RED`
- `ORTH`
  
Please refer to paper "Optimization and Identification of Lattice Quantizers" for the detailed definition of each function.

**The implementation of function above has been checked to be correct.**

In `ilc.py` is the implementation of iterative lattice construction algorithm from the original paper. Directly modified from the pseudocode. 

**In `draw_theta_image.py` is a code to draw the theta image of lattice, but it still has some bugs to be FIXED.**

### Unit test and visualization

In `test_2d.py` is the test code for a 2d-case visualization. It can output a hexagonal lattice as expected. 


In `test_3d.py` is the test code for a 3d-case visualization. It can output a body-centered cubic lattice. Note that in this case we performed rotation for a better visualization.

2d and 3d cases are all correct as expected. Visulization results are as follows. You can also generate these results by directly running `test_2d.py` and `test_3d.py`.

- 2d case

  
  Basis vectors are plotted in red and blue respectively. Lattice points are black dots. It is clear that our implementation generates a hexagonal lattice.
  ![2d](figures/2d.png)

- 3d case

  
  Basis vectors are plotted in red, blue and green respectively. Lattice points are black dots. It is clear that our implementation generates a body-centered cubic lattice.

  Below are 3d visulization results looking from x,y,z axes respectively.
  ![3d](figures/3d_look_from_x.png)
  ![3d](figures/3d_look_from_y.png)
  ![3d](figures/3d_look_from_z.png)

  Below is a gif demo of our 3d visulization.
  ![3d](figures/3d_rotation_up_down.gif)
