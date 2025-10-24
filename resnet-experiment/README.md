# Inline Stab-SGD: A practical implementation of Stab-SGD for ResNets.

This code provides the practical implementation of Alg. 1 presented in "Stab-SGD: Noise-Adaptivity in Smooth
Optimization with Stability Ratios" by David A. R. Robin, Killian Bakong and Kevin Scaman [3].

It also features code to test it on a ResNet-56 network [1], using the CIFAR-10 dataset [2].

## Setup

Install the required packages thanks to environment.yml

## Run code

To run the code, one may use a command like:

```
python main.py --seed 1 --batch-size 128 --lr 10. --min-gd-iterations 64000 --min-total-iterations 128000 --weight-decay 1e-4 --zeta-start 100 --zeta 100 --kappa 0.1 --gamma 1 --output-folder "results"
```

where the parameters are:
- seed: Random seed
- batch-size: Batch-size (default: 128)
- lr: Learning rate (default: 10.)
- min-gd-iterations: Minimal amount of gradient descent iterations (i.e. the Stability Ratio compute iterations are not counted, default: 64 000)
- min-total-iterations: Minimal amount of total iterations (i.e. including Stability Ratio computation iterations, default: 128 000)
- weight-decay: Weight-decay (default: 1e-4)
- zeta-start: Zeta parameter (see [3]) for the initial computation of the Stability Ratio (default: 100.)
- zeta: Zeta parameter (see [3])
- kappa: Kappa parameter (see [3])
- gamma: Gamma parameter (see [3])
- output-folder: Folder to store the results


Note that the training will stop once the two conditions on the minimal amount of iterations are fulfilled.

## Results saved by default

The results saved by default are the accuracies and losses of both training and testing datasets.
Those values are computed every $\texttt{len(train-dataloader)}$ total iterations _and_ gradient descent only interations, which corresponds to the number of iterations in an epoch.
Moreover, the optimizer "info" is stored in a .json file, which, in particular, contains the history of Stability Ratio and kurtosis values, as well as the total number of gradient samples used to compute them throughout the training.



## References

- [1] K. He, X. Zhang, S. Ren, J. Sun. *Deep Residual Learning for Image Recognition.* CVPR 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)  
- [2] A. Krizhevsky. *Learning Multiple Layers of Features from Tiny Images.* Technical Report, 2009. (CIFAR-10 dataset)
- [3] D. A. R. Robin, K. Bakong, K. Scaman. *Stab-SGD: Noise-Adaptivity in Smooth Optimization with Stability Ratios*. Neurips 2025.
