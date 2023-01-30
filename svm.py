import numpy as np

class SVM:

    def __init__(self, learning_rate=0.001, _lambda=0.01):
        self.lr = learning_rate
        self._lambda = _lambda
        self.w = None
        self.b = None

    def calculate(self, X, Y):
        n_class = X.shape[1]
        new_Y=[]
        for i in Y:
            if i <=0:
                new_Y.append(-1)
            else:
                new_Y.append(1)

        zeros=[]
        for i in range(n_class):
            zeros.append(0.)
        self.w=np.array(zeros)
        self.b = 0

        for i in range(1000):
            for idx, value in enumerate(X):
                if new_Y[idx] * (np.dot(value, self.w) - self.b) >= 1 :
                    self.w -= self.lr * 2 * self._lambda * self.w
                else:
                    self.w -= self.lr * (2 * self._lambda * self.w - np.dot(value, new_Y[idx]))
                    self.b -= self.lr * new_Y[idx]

    def get_my_Y(self, X):
        arr = np.dot(X, self.w) - self.b
        new_arr=[]
        for i in arr:
            if i>0:
                new_arr.append(1)
            if i==0:
                new_arr.append(0)
            if i<0:
                new_arr.append(-1)        
        return np.array(new_arr)