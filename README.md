# The Role of Depth, Width, and Tree Size in Expressiveness of Deep Forest

This is the code of simulation and benchmark experiments for ECAI 2024 paper "[The Role of Depth, Width, and Tree Size in Expressiveness of Deep Forest](https://lyushenhuan.github.io/papers/lyu2024role.pdf)" by Shen-Huan Lyu, Jin-Hui Wu, Qin-Cheng Zheng, and Baoliu Ye. In the paper, we theoretically prove that depth can exponentially enhance the expressiveness of deep forests compared with width and tree size.

## Getting Started

The packages used are shown below.

```
conda create -n deepTree python=3.9
conda activate deepTree
pip install matplotlib
pip install scipy==1.13
pip install scikit-learn==1.4.2
```

## Examples

1. Obtain the test accuracy of learning parity functions using trees, deep trees of different depths, and random forests of different numbers of trees.

   ```
   python example/dataForDeepTree.py --input_dimension 2 --data_number 1000000
   python example/simulation.py --input_dimension 2 --data_number 1000000
   ```

   The argument input_dimension can be changed to 4 and 8 to obtain rest simulation results.

2. Plot simulation results after running the above simulations (Figure 8 in the paper).

   ```
   python example/plot.py
   ```

3. Obtain the test accuracy of benchmark datasets using deep forests of different numbers of trees and tree sizes.

   ```
   python example/demo.py 
   ```

4. Plot simulation results after running the above benchmarks (Figure 9 in the paper).

   ```
   python example/plotbar.py
   ```

## Cite the Paper

```
@inproceedings{lyu2024role,
  title={The Role of Depth, Width, and Tree Size in Expressiveness of Deep Forest},
  author={Lyu, Shen-Huan and Wu, Jin-Hui and Zheng, Qin-Cheng and Ye, Baoliu},
  booktitle={Proceedings of the 27th European Conference on Artifical Intelligence},
  pages={2042--2049},
  year={2024}
}
```

