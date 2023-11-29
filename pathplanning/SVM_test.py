import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from scipy.io import loadmat
from sklearn import svm
def generate_data(rho,O):
    X=[]
    Y=[]
    for i in range(len(rho)):
        X.append(rho[i])
        Y.append(-1)
    for i in range(len(O)):
        X.append(O[i])
        Y.append(1)
    raw_data = {
    'X': X,
    'y': Y
    }
    data = pd.DataFrame(raw_data.get('X'), columns=['X1', 'X2'])
    data['y'] = raw_data.get('y')
    print(data)
    return data
        
def plot_init_data(data, fig, ax):
    positive = data[data['y'].isin([1])]
    negative = data[data['y'].isin([-1])]

    ax.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
    ax.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')

def find_decision_boundary(svc, x1min, x1max, x2min, x2max, diff):
    x1 = np.linspace(x1min, x1max, 1000)
    x2 = np.linspace(x2min, x2max, 1000)

    cordinates = [(x, y) for x in x1 for y in x2]
    x_cord, y_cord = zip(*cordinates)
    c_val = pd.DataFrame({'x1':x_cord, 'x2':y_cord})
    c_val['cval'] = svc.decision_function(c_val[['x1', 'x2']])

    decision = c_val[np.abs(c_val['cval']) < diff]
    
    return decision.x1, decision.x2


rho=[[-50,0],[-50,-50],[0,-80],[50,-40],[50,10]]
O=[[-200,150],[-20,120],[20,200]]
data=generate_data(rho,O)
# fig, ax = plt.subplots(figsize=(12,8))
# plot_init_data(data, fig, ax)
# ax.legend()
# plt.show()

svc = svm.LinearSVC(C= 0.1,loss='hinge', max_iter=1000)

svc.fit(data[['X1', 'X2']], data['y'])
# Get the coefficients and intercept of the decision hyperplane
coef = svc.coef_[0]
intercept = svc.intercept_[0]

# Calculate the margin
margin = 2 / np.linalg.norm(coef)
print("Margin:", margin)
# x1, x2 = find_decision_boundary(svc, -200, 250, -200, 250, 1 * 10**-2)
# fig, ax = plt.subplots(figsize=(12,8))
# ax.scatter(x1, x2, s=10, c='r',label='Boundary')
# plot_init_data(data, fig, ax)
# ax.set_title('SVM (C=1) Decision Boundary')
# ax.legend()
# plt.show()

# plot the decision function for each datapoint on the grid
xx, yy = np.meshgrid(np.linspace(-250, 250, 500), np.linspace(-250, 250, 500))
np.random.seed(0)
Z = svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(10,10))
plt.imshow(
    Z,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    aspect="auto",
    origin="lower",
    cmap=plt.cm.PuOr_r,
)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2, linestyles="dashed")
positive = data[data['y'].isin([1])]
negative = data[data['y'].isin([-1])]

plt.scatter(positive['X1'], positive['X2'], s=50, marker='x', label='Positive')
plt.scatter(negative['X1'], negative['X2'], s=50, marker='o', label='Negative')
# plt.scatter(positive['X1'], X[:, 1], s=30, c=y, cmap=plt.cm.Paired, edgecolors="k")

# plt.axis([-3, 3, -3, 3])
plt.title('Linear SVM classifier using polynomial features')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()