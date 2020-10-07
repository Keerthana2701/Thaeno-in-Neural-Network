# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 12:14:25 2020

@author: Vikee
"""
# adding two scalars


from theano import *
import theano.tensor as T
import theano
import numpy


x = tensor.dscalar()
y = tensor.dscalar()

z = x + y
f = theano.function([x,y], z) # thaeno function with i/p x and y and o/p z
print(f(1.5, 2.5))



    
    
    
    
