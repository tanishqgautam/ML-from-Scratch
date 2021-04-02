import numpy as np

class LinearRegression:
    def __init__(self):
        self.learning_rate = 0.01
        self.epochs = 10000
        
    def y_hat(self, X, w):
        return np.dot(w.T, X)

    def cost(self, yhat, y):
        C = 1 / self.m * np.sum(np.power(yhat - y, 2))
        return C

    def gradient_descent(self, w, X, y, yhat):
        dCdW = 2 / self.m * np.dot(X, (yhat - y).T)
        w = w - self.learning_rate * dCdW
        return w

    def run(self, X, y):
        # Add x1 = 1
        ones = np.ones((1, X.shape[1]))
        X = np.append(ones, X, axis=0)

        self.m = X.shape[1]
        self.n = X.shape[0]

        w = np.zeros((self.n, 1))

        for epoch in range(self.epochs + 1):
            yhat = self.y_hat(X, w)
            cost = self.cost(yhat, y)

            if epoch % 1000 == 0:
                print(f"Cost at iteration {epoch} is {cost}")

            w = self.gradient_descent(w, X, y, yhat)

        return w



if __name__ == "__main__":
    X = np.random.rand(1, 500)
    y = 3 * X + 5 + np.random.randn(1, 500) * 0.1
    regression = LinearRegression()
    w = regression.run(X, y)