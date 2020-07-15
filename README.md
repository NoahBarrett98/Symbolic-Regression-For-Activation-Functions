# Symbolic-Regression-For-Activation-Functions
Using Symbolic Regression to discover useful activation functions for neural networks
This is an simple attempt at using symbolic regression to find activation functions which result in the most loss. It lacks in many ways.

1) it may be more useful to look at how "active" the activation function is, in that it is producing meaningful and variant activations

2) This uses out of the box implmentations of both GP and Neural nets, this is highly un optimized.

The idea is to take a very simple net, use symbolic regression to find useful activations for the particular dataset. These activations can hopefully extrapolate to larger networks which require much more time for training, etc.
