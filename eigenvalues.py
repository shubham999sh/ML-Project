import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from numpy import array
from numpy.linalg import eig
from numpy.linalg import inv
from numpy import dot
from numpy import diag

#define matrix

A = array([[1,2,3],[4,5,6],[7,8,9]])
print(A)

# Eigen Decompostion

values, vectors = eig(A)
'''print(values)
print(vectors)'''

#confirm it is decompoistion of eigen value

B = A.dot(vectors[:,0])
print(B)

#C = vectors[:,0] * values[0]
#print (C)

# reconstruct the matrix

Q = vectors
R = inv(Q)
L = diag(values)
B = Q.dot(L).dot(R)
print(B)