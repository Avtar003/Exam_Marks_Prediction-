import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_csv("D:\yadav\Coding Blocks\Machine Learning\Challenge - Hardwork Pays off\Train\Linear_X_Train.csv")
y = pd.read_csv("D:\yadav\Coding Blocks\Machine Learning\Challenge - Hardwork Pays off\Train\Linear_Y_Train.csv")


X = X.values
y = y.values

u = X.mean()
std = X.std()

X = (X-u)/std

# plt.style.use('seaborn')
# plt.scatter(X,Y,color='red')
# plt.show()

def hypothesis(x,theta):
    y_ = theta[0]+theta[1]*x
    return y_

def Gradient(X,Y,theta):
    m = X.shape[0]
    grad = np.zeros((2,))
    for i in range(m):
        x = X[i]
        y_ = hypothesis(x,theta)
        y = Y[i]
        grad[0] += (y_ - y)
        grad[1] += (y_ - y)*x
    return grad/m

def error(X,Y,theta):
    m = X.shape[0]
    total_error = 0.0
    for i in range(m):
        y_ = hypothesis(X[i],theta)
        total_error += (y_ - Y[i])**2
    return total_error/m

def GradientDescent(X,Y,max_step=100,learning_rate=0.1):
    theta = np.zeros((2,))
    error_list = []
    theta_list = []
    for i in range(max_step):
        grad = Gradient(X,y,theta)
        e = error(X,Y,theta)
        error_list.append(e)
        theta[0] = theta[0]-learning_rate*grad[0]
        theta[1] = theta[1]-learning_rate*grad[1]
        theta_list.append((theta[0],theta[1]))
    return theta,error_list,theta_list


theta,errorlist,theta_list = GradientDescent(X,y)
#
#
y_ = hypothesis(X,theta)
#
# plt.scatter(X,y)
# plt.plot(X,y_,color='black',label='Prediction')
# plt.legend()
# plt.show()


X_test = pd.read_csv("D:\yadav\Coding Blocks\Machine Learning\Challenge - Hardwork Pays off\Test\Linear_X_Test.csv").values
y_test = hypothesis(X_test,theta)


def r2_score(Y,Y_):
    num = np.sum(((Y-Y_)**2))
    den = np.sum((Y-Y.mean())**2)
    r2 = (1-num/den)
    return r2*100


T0 = np.arange(-40,40,1)
T1 = np.arange(40,120,1)

T0,T1 = np.meshgrid(T0,T1)

J = np.zeros(T0.shape)
for i in range(J.shape[0]):
    for j in range(J.shape[1]):
        y_ = T1[i,j]*X+T0[i,j]
        J[i,j] = np.sum((y-y_)**2)/y.shape[0]


errorlist = np.array(errorlist)
fig = plt.figure()
axes = fig.gca(projection="3d")
axes.plot_surface(T0,T1,J,cmap='rainbow')
axes.scatter(theta_list[:,0],theta_list[:,1],*errorlist)
plt.show()

theta_list = np.array(theta_list)
#
# plt.plot(theta_list[:,0],label='Theta0')
# plt.plot(theta_list[:,1],label='Theta1')
# plt.legend()
# plt.show()