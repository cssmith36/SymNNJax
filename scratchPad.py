import numpy as np
import minresQLP2 as MQLP
import torch
import jax as jnp
from operator import add
#import RBM

#print(list(RBM.Test.parameters()))

a = np.array([1,2,3])
b = np.array([4,5,6])

print(np.kron(a,a))


a = np.array([i for i in range(5)])
b = np.array([5,5,5,5,5])

c = b+b
d = b+b+b
e = b+b+b+b
f = np.array([0,0,0,0,0])

a = np.array([a,a,a,a])
q = np.array([f,b,c,d])
XX = a+q
X = XX.flatten()
XX = XX.reshape((5,4))

print(XX)


#a = a.flatten()
#print(a.reshape((2,3)))
#print(a.flatten())
#a = np.concatenate(([a],[b]), axis = 0)
#q = jnp.numpy.multiply(a,b)

#print(q)
#print(q[0])

'''https://github.com/pascanur/theano_optimize'''
'''
a = np.array([1,1,1,-1])

A = np.outer(a,a)
print(A)
b = np.array([3,5,9,3])

AFun = lambda x: ([np.dot(A, x)])

print(type(AFun))

X = MQLP.MinresQLP(AFun,b,1e-6,100)
print(X)

inv = np.linalg.pinv(A)
print(inv)
print(np.matmul(inv,b))'''