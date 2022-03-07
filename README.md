# TUSLA_RELU

This repository is the official implementation of "Non-asymptotic estimates for TUSLA algorithm for non-convex learning with applications to neural networks with ReLU activation function" (https://arxiv.org/abs/2107.08649).

Abstract
We consider non-convex stochastic optimization problems where the objective functions have super-linearly growing and discontinuous stochastic gradients. In such a setting, we provide a nonasymptotic analysis for the tamed unadjusted stochastic Langevin algorithm (TUSLA) introduced in Lovas et al. (2021). In particular, we establish non-asymptotic error bounds for the TUSLA algorithm in Wasserstein1 and Wasserstein-2 distances. The latter result enables us to further derive non-asymptotic estimates for the expected excess risk. To illustrate the applicability of the main results, we consider an example from transfer learning with ReLU neural networks, which represents a key paradigm in machine learning. Numerical experiments are presented for the aforementioned example which supports our theoretical findings. Hence, in this setting, we demonstrate both theoretically and numerically that the TUSLA algorithm can solve the optimization problem involving neural networks with ReLU activation function. Besides, we provide simulation results for synthetic examples where popular algorithms, e.g. ADAM, AMSGrad, RMSProp, and (vanilla) SGD, may fail to find the minimizer of the objective functions due to the super-linear growth and the discontinuity of the corresponding stochastic gradient, while the TUSLA algorithm converges rapidly to the optimal solution.

## Dependencies

- Python 3.6
- Pytorch 1.8.0 + cuda
- scikit-learn

## Training

To train the one-dimensional optimization problems, transfer learning, and real-world applications in the paper, run the following commands:

### Artifical examples (Section 3.2.)
```train
python 1D_optimization_beta.py --lr 1e-3 --epochs 1000 --eta 0 --beta 1e10 --r 14
python 1D_optimization_normal.py --lr 0.002 --epochs 500 --eta 0 --beta 1e10 --r 14
```

### Feed-forward neural network with fixed input weights (Section 3.1.)
```train
python transfer_learning_singleNN.py --lr 0.5 --eta 1e-6 --beta 1e10
```

### Fashion MNIST
```train
## SLFN

python main.py --model slfn --optimizer tusla --lr 0.5 --r .5 --beta 1e12 --total_epoch 200 --seed 111 --eta 1e-5 
python main.py --model slfn --optimizer adam --lr 0.001 --total_epoch 200 --seed 111 --weight_decay 1e-5
python main.py --model slfn --optimizer amsgrad --lr 0.001 --total_epoch 200 --seed 111 --weight_decay 1e-5
python main.py --model slfn --optimizer rmsprop --lr 0.001 --total_epoch 200 --seed 111 --weight_decay 1e-5

## TLFN
python main.py --model tlfn --optimizer tusla --lr 0.5 --r .5 --beta 1e12 --total_epoch 200 --seed 111 --eta 1e-5 
python main.py --model tlfn --optimizer adam --lr 0.001 --total_epoch 200 --seed 111 --weight_decay 1e-5
python main.py --model tlfn --optimizer amsgrad --lr 0.001 --total_epoch 200 --seed 111 --weight_decay 1e-5
python main.py --model tlfn --optimizer rmsprop --lr 0.001 --total_epoch 200 --seed 111 --weight_decay 1e-5
```
Please use the jupyter notebook ``visualization.ipynb'' to visualize the test curves of different optimizers. We also provide logs for the models in the main paper. 

### Concrete
```train
python main.py --seed 111 --optimizer tusla --lr 0.5 --r 0.5 --beta 1e12 --dataset concrete --batch_size 256
python main.py --seed 222 --optimizer tusla --lr 0.5 --r 0.5 --beta 1e12 --dataset concrete --batch_size 256
python main.py --seed 333 --optimizer tusla --lr 0.5 --r 0.5 --beta 1e12 --dataset concrete --batch_size 256 

python main.py --seed 111 --optimizer adam --lr 0.001 --dataset concrete --batch_size 256
python main.py --seed 222 --optimizer adam --lr 0.001 --dataset concrete --batch_size 256
python main.py --seed 333 --optimizer adam --lr 0.001 --dataset concrete --batch_size 256 

python main.py --seed 111 --optimizer amsgrad --lr 0.001 --dataset concrete --batch_size 256
python main.py --seed 222 --optimizer amsgrad --lr 0.001 --dataset concrete --batch_size 256
python main.py --seed 333 --optimizer amsgrad --lr 0.001 --dataset concrete --batch_size 256 

python main.py --seed 111 --optimizer rmsprop --lr 0.001 --dataset concrete --batch_size 256
python main.py --seed 222 --optimizer rmsprop --lr 0.001 --dataset concrete --batch_size 256
python main.py --seed 333 --optimizer rmsprop --lr 0.001 --dataset concrete --batch_size 256
```
Please use the jupyter notebook ``visualization.ipynb'' to visualize the test curves of different optimizers. We also provide logs for the models in the main paper. 


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
