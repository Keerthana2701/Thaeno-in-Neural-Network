# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 16:04:20 2020

@author: Vikee
"""


import theano
import theano.tensor as T
from theano.ifelse import ifelse
import numpy as np
from random import random
import matplotlib.pyplot as plt


#Define variables:

# we have 3 weight vectors corresponding to each neuron and 2 bias units corresponding to 2 layers.


x = T.matrix('x')
w1 = theano.shared(np.array([random(),random()]))
w2 = theano.shared(np.array([random(),random()]))
w3 = theano.shared(np.array([random(),random()]))
b1 = theano.shared(1.)
b2 = theano.shared(1.)
learning_rate = 0.01

a1 = 1/(1+T.exp(-T.dot(x,w1)-b1))  # sigmoid activation fn 
a2 = 1/(1+T.exp(-T.dot(x,w2)-b1))
x2 = T.stack([a1,a2],axis=1) # outputs of a1 and a2 to be combined into a matrix whose dot product can be taken with the weights vector.
a3 = 1/(1+T.exp(-T.dot(x2,w3)-b2))

# Define gradient and update rule

# determine the gradients of 3 weight vectors and 2 bias units and update them accordingly.

a_hat = T.vector('a_hat') #Actual output
cost = -(a_hat*T.log(a3) + (1-a_hat)*T.log(1-a3)).sum()
dw1,dw2,dw3,db1,db2 = T.grad(cost,[w1,w2,w3,b1,b2])

# train function does the weight updation

train = theano.function(inputs = [x,a_hat],
    outputs = [a3,cost],
    updates = [
        [w1, w1-learning_rate*dw1],
        [w2, w2-learning_rate*dw2],
        [w3, w3-learning_rate*dw3],
        [b1, b1-learning_rate*db1],
        [b2, b2-learning_rate*db2]
              ])

# Train the model

inputs = [
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
]
outputs = [1,0,0,1]

#Iterate through all inputs and find outputs:

cost = []
for iteration in range(30000):
    pred, cost_iter = train(inputs, outputs)
    cost.append(cost_iter)
    
#Print the outputs:
print ('The outputs of the NN are:')
for i in range(len(inputs)):
    print ('The output for x1=%d | x2=%d is %.2f' % (inputs[i][0],inputs[i][1],pred[i]))
    
#Plot the flow of cost:
print ('\nThe flow of cost during model run is as following:')

plt.plot(cost)