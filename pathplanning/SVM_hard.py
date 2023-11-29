import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cvxopt.solvers

def get_training_examples():
    X1 = np.array([[-50,0],[-50,-50],[0,-80],[50,-40],[50,10]])
    y1 = np.ones(len(X1))
    X2 = np.array([[-200,150],[-20,120],[20,200]])
    y2 = np.ones(len(X2)) * -1
    return X1, y1, X2, y2

def get_dataset(get_examples):
    X1, y1, X2, y2 = get_examples()
    X, y = get_dataset_for(X1, y1, X2, y2)
    return X, y

def get_dataset_for(X1, y1, X2, y2):
    X = np.vstack((X1, X2))
    y = np.hstack((y1, y2))
    return X, y


# computing 'w'
def compute_w(multipliers, X, y):
    return np.sum(multipliers[i] * y[i] * X[i] for i in range(len(y))) 

# compute 'b'
def compute_b(w, X, y):
    return np.sum([y[i] - np.dot(w, X[i]) for i in range(len(X))])/len(X)

def find_decision_boundary(w,b, x1min, x1max, x2min, x2max, diff):
    x1 = np.linspace(x1min, x1max, 1000)
    x2 = np.linspace(x2min, x2max, 1000)

    cordinates = [(x, y) for x in x1 for y in x2]
    x_cord, y_cord = zip(*cordinates)
    c_val = pd.DataFrame({'x1':x_cord, 'x2':y_cord})
    c_val['cval'] = c_val['x1']*w[0]+c_val['x2']*w[1]+b-1

    decision = c_val[np.abs(c_val['cval']) < diff]
    
    return decision.x1, decision.x2


X, y = get_dataset(get_training_examples) 
m = X.shape[0] 

# Gram matrix - The matrix of all possible inner products of X. 
K = np.array([np.dot(X[i], X[j])
              for i in range(m)
              for j in range(m)]).reshape((m, m)) 
P = cvxopt.matrix(np.outer(y, y) * K) 
q = cvxopt.matrix(-1 * np.ones(m)) 

# let alpha be 1 for all obervations
# Equality constraints 
A = cvxopt.matrix(y, (1, m)) # create 1*m matrix of y 
b = cvxopt.matrix(0.0) 

# Inequality constraints 
G = cvxopt.matrix(np.diag(-1 * np.ones(m))) 
h = cvxopt.matrix(np.zeros(m)) 

# Solve the problem 
solution = cvxopt.solvers.qp(P, q, G, h, A, b) # return # dictionary which contains iterators

# Lagrange multipliers 
multipliers = np.ravel(solution['x']) 
sorted(multipliers)

# Support vectors have positive multipliers. 
has_positive_multiplier = multipliers > 1e-07 
sv_multipliers = multipliers[has_positive_multiplier] 
 
support_vectors = X[has_positive_multiplier] 
support_vectors_y = y[has_positive_multiplier] 

w = compute_w(multipliers, X, y) 
w_from_sv = compute_w(sv_multipliers, support_vectors, support_vectors_y)
 
# print(w)          
print(w_from_sv)  

margin = 2 / np.linalg.norm(w_from_sv)
print(margin)

b = compute_b(w, support_vectors, support_vectors_y)  


qqq = pd.DataFrame(X)
qqq['y']=y
qqq[qqq['y']==-1][0]
qqq[qqq['y']==-1][1]
x1, x2 = find_decision_boundary(w,b, -200, 250, -200, 250, 1 * 10**-2)
plt.scatter(x1, x2, s=10, c='r',label='Boundary')
plt.scatter(qqq[qqq['y']==1][0],qqq[qqq['y']==1][1])
plt.scatter(qqq[qqq['y']==-1][0],qqq[qqq['y']==-1][1])
plt.show()

