#calulate beta
import gurobipy as gp
import numpy as np
import time
from numpy import linalg as LA

def get_alpha(p1,p2,xm):
    M=np.zeros((2,2))
    M[:,0]=p1-xm
    M[:,1]=p2-xm
    I=np.ones((1,2))
    a=np.dot(I,np.linalg.inv(M))

    return np.transpose(a)

rho=np.array([[-50,0],[-50,-50],[0,-80],[50,-50],[50,0]])
O=np.array([[-200,150],[-20,120],[20,200]])
Xm=(rho[0,:]+rho[-1,:])/2
a=[]
for i in range(rho.shape[0]-1):
    a.append(get_alpha(np.transpose(rho[i,:]),np.transpose(rho[i+1,:]),np.transpose(Xm)))

# 创建优化问题
model = gp.Model()

# 创建变量

# 创建多个01整数变量
num_variables = len(a) # 变量数量
variables = [model.addVar(lb=0,ub=1,vtype=gp.GRB.INTEGER, name='x{}'.format(i)) for i in range(num_variables)] # 创建包含5个元素的0-1整数变量x
beta=model.addVar(vtype=gp.GRB.CONTINUOUS,name='beta')
# x=model.addVar(lb=0,ub=1,vtype=gp.GRB.INTEGER,name='x')
# 添加约束条件
# 添加约束条件：变量之和等于1
# constraint_expr = gp.LinExpr()  # 创建线性表达式
# for var in variables:
#     constraint_expr.add(var)  # 将变量添加到线性表达式中
# model.addConstr(constraint_expr == 1, name='constraint')  # 添加约束条件
# 添加约束条件
# for  i, var in enumerate(variables):
#     for j in range(rho.shape[0]):
#         model.addConstr(var*(np.dot(np.transpose(a[i]),(np.transpose(rho[j,:])-np.transpose(Xm)))-1)<=0)
#     for j in range(O.shape[0]):
#         model.addConstr(0<=var*(np.dot(np.transpose(a[i]),(np.transpose(O[j,:])-np.transpose(Xm)))-beta))
# 求解优化问题
for j in range(rho.shape[0]):
    expr = gp.LinExpr()
    expr.add(np.dot(np.transpose(a[3]),(np.transpose(rho[j,:])-np.transpose(Xm))))
    model.addConstr(expr<=beta)
for j in range(O.shape[0]):
    expr = gp.LinExpr()
    expr.add(np.dot(np.transpose(a[3]),(np.transpose(O[j,:])-np.transpose(Xm))))
    model.addConstr(expr>=beta)

model.setObjective(beta, sense=gp.GRB.MAXIMIZE)
model.optimize()  # 使用CBC求解器（支持整数变量）

# 获取求解状态
status = model.status

# 判断求解是否成功
if status == gp.GRB.OPTIMAL:
    obj_value = model.objVal
    print(obj_value)
    # ... 处理最优解 ...
else:
    print("The model is infeasible.")
    # ... 处理不可行情况 ...


