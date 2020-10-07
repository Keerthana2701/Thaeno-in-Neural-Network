# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:30:11 2020

@author: Vikee
"""
# The script iteratively modifies the first vector in the previous example, using gradient descent, such that the dot product would have value 20.
# It should print the value at each of the 10 iterations:

from theano import *
import theano.tensor as T
import theano
import numpy

x = theano.tensor.fvector('x')
target = theano.tensor.fscalar('target')

W = theano.shared(numpy.asarray([0.2, 0.7]), 'W')
y = (x * W).sum()

cost = theano.tensor.sqr(target - y)
gradients = theano.tensor.grad(cost, [W])
W_updated = W - (0.1 * gradients[0])
updates = [(W, W_updated)]

f = theano.function([x, target], y, updates=updates)

for i in range(10):
    output = f([1.0, 1.0], 20.0)
    print (output)
    