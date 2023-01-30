from sys import api_version
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from svm import SVM

X, Y = datasets.make_blobs(n_samples=60, n_features=2, centers=2, cluster_std=0.65, random_state=40)

y=[]
for i in Y:
    if i==0:
        y.append(-1)
    else:
        y.append(1)
Y = np.array(y)            

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

my_tree = SVM()
my_tree.calculate(X_train, Y_train)
my_Y = my_tree.get_my_Y(X_test)


def possibly(Y_test, my_Y):
    sum=0
    n=len(Y_test)
    for i in range(n):
        if Y_test[i] == my_Y[i]:
            sum+=1
    return sum/n    

poss = possibly(Y_test, my_Y)
print("Precision is :" , poss)

def get_value(x, w, b, offset):
    return (-w[0] * x + b + offset) / w[1]

def set_figure():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    plt.scatter(X[:, 0], X[:, 1], marker="o", c=Y)  
    
    x_min = np.amin(X[:, 0])
    x_max = np.amax(X[:, 0])

    y_min_0 = get_value(x_min, my_tree.w, my_tree.b, 0)
    y_max_0 = get_value(x_max, my_tree.w, my_tree.b, 0)

    y_min_negetive_1 = get_value(x_min, my_tree.w, my_tree.b, -1)
    y_max_negetive_1 = get_value(x_max, my_tree.w, my_tree.b, -1)

    y_min_1 = get_value(x_min, my_tree.w, my_tree.b, 1)
    y_max_1 = get_value(x_max, my_tree.w, my_tree.b, 1)

    ax.plot([x_min, x_max], [y_min_negetive_1, y_max_negetive_1], "k--")
    ax.plot([x_min, x_max], [y_min_0, y_max_0], "r")   #(x_min,y_min_0) , (x_max,y_max_0)
    ax.plot([x_min, x_max], [y_min_1, y_max_1], "k--")

    x1_min = np.amin(X[:, 1])
    x1_max = np.amax(X[:, 1])
    ax.set_ylim([x1_min - 2, x1_max + 2])

    plt.show()

set_figure()