import numpy as np
from sklearn.datasets import make_blobs


class LogisticRegression:
    def __init__(self, X):
        self.lr = 0.1
        self.epochs = 10000
        self.m, self.n = X.shape
        self.weights = np.zeros((self.n, 1))
        self.bias = 0

    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def cost(self,y_predict,y):
        return (-1 / self.m * np.sum(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict)))


    def gradient(self, y_predict,y):
        dw = 1 / self.m * np.dot(X.T, (y_predict - y))
        db = 1 / self.m * np.sum(y_predict - y)
        return dw, db

    def run(self, X, y):
        for epoch in range(self.epochs + 1):
            
            y_predict = self.sigmoid(np.dot(X, self.weights) + self.bias)
            cost = self.cost(y_predict,y)
            dw, db = self.gradient(y_predict,y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if epoch % 1000 == 0:
                print(f"Cost after iteration {epoch}: {cost}")

        return self.weights, self.bias

    def predict(self, X):
        y_predict = self.sigmoid(np.dot(X, self.weights) + self.bias)
        y_predict_labels = y_predict > 0.5

        return y_predict_labels
  

if __name__ == "__main__":
    np.random.seed(1)
    X, y = make_blobs(n_samples=1000, centers=2)
    y = y[:, np.newaxis]
    logreg = LogisticRegression(X)
    w, b = logreg.run(X, y)
    y_predict = logreg.predict(X)

    print(f"Accuracy: {np.sum(y==y_predict)/X.shape[0]}")
