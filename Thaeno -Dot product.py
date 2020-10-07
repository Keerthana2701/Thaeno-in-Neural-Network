# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:29:08 2020

@author: Vikee
"""

# It calculates the dot product between vectors [0.2, 0.9] and [1.0, 1.0].


import theano
import numpy

x = theano.tensor.fvector('x')
W = theano.shared(numpy.asarray([0.2, 0.7]), 'W')
y = (x * W).sum()

f = theano.function([x], y)

output = f([1.0, 1.0])
print (output)