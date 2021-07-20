# TUSLA_RELU

This repository is the official implementation of "Non-asymptotic estimates for TUSLA algorithm for non-convex learning with applications to neural networks with ReLU activation function" (https://arxiv.org/abs/2107.08649).

Abstract
We consider non-convex stochastic optimization problems where the objective functions have super-linearly growing and discontinuous stochastic gradients. In such a setting, we provide a nonasymptotic analysis for the tamed unadjusted stochastic Langevin algorithm (TUSLA) introduced in Lovas et al. (2021). In particular, we establish non-asymptotic error bounds for the TUSLA algorithm in Wasserstein1 and Wasserstein-2 distances. The latter result enables us to further derive non-asymptotic estimates for the expected excess risk. To illustrate the applicability of the main results, we consider an example from transfer learning with ReLU neural networks, which represents a key paradigm in machine learning. Numerical experiments are presented for the aforementioned example which supports our theoretical findings. Hence, in this setting, we demonstrate both theoretically and numerically that the TUSLA algorithm can solve the optimization problem involving neural networks with ReLU activation function. Besides, we provide simulation results for synthetic examples where popular algorithms, e.g. ADAM, AMSGrad, RMSProp, and (vanilla) SGD, may fail to find the minimizer of the objective functions due to the super-linear growth and the discontinuity of the corresponding stochastic gradient, while the TUSLA algorithm converges rapidly to the optimal solution.

## Dependencies

- Python 3.6
- Pytorch 1.8.0 + cuda
- scikit-learn

## Training

To train the one-dimensional optimization problems and transfer learning in the paper, run the following commands:

### Artifical example(Section 3.2.)
```train
python 1D_optimization_beta.py --lr 1e-3 --epochs 1000 --eta 0 --beta 1e10 --r 14
python 1D_optimization_normal.py --lr 0.002 --epochs 500 --eta 0 --beta 1e10 --r 14
```

### Feed-forward neural network with fixed input weights.(Section 3.1.)
```train
python transfer_learning_singleNN.py --lr 0.5 --eta 1e-25 --beta 1e10
```

## Citing
If you use this codebase in your work, please cite:
```
@article{tusla_relu,
  title={NON-ASYMPTOTIC ESTIMATES FOR TUSLA ALGORITHM FOR NON-CONVEX LEARNING WITH APPLICATIONS TO NEURAL NETWORKS WITH RELU ACTIVATION FUNCTION},
  author={D.Y. Lim and A. Neufeld and S. Sabanis and Y. Zhang},
  journal={arXiv:2107.08649},
  year={2021},  
}
```
