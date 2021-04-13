import numpy as np
import cvxopt
import matplotlib.pyplot as plt

def plot_contour(X, y, svm):
    h = 0.01
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    points = np.c_[xx.ravel(), yy.ravel()]

    Z = svm.predict(points)
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.show()


def gaussian(x, z, sigma=0.1):
    return np.exp(-np.linalg.norm(x - z, axis=1) ** 2 / (2 * (sigma ** 2)))


class SVM:
    def __init__(self, kernel=gaussian, C=1):
        self.kernel = kernel
        self.C = C

    def solveQp(self,m):
        P = cvxopt.matrix(np.outer(y, y) * self.K)
        q = cvxopt.matrix(-np.ones((m, 1)))
        G = cvxopt.matrix(np.vstack((np.eye(m) * -1, np.eye(m))))
        h = cvxopt.matrix(np.hstack((np.zeros(m), np.ones(m) * self.C)))
        A = cvxopt.matrix(y, (1, m), "d")
        b = cvxopt.matrix(np.zeros(1))
        cvxopt.solvers.options["show_progress"] = False
        return cvxopt.solvers.qp(P, q, G, h, A, b)


    def fit(self, X, y):
        self.y = y
        self.X = X
        m, n = X.shape
        self.K = np.zeros((m, m))
        for i in range(m):
            self.K[i, :] = self.kernel(X[i, np.newaxis], self.X)

        sol = self.solveQp(m)
        self.alphas = np.array(sol["x"])

    def predict(self, X):
        y_predict = np.zeros((X.shape[0]))
        sv = self.fetch_parameters(self.alphas)

        for i in range(X.shape[0]):
            y_predict[i] = np.sum( self.alphas[sv] * self.y[sv, np.newaxis] * self.kernel(X[i], self.X[sv])[:, np.newaxis])

        return np.sign(y_predict + self.b)

    def fetch_parameters(self, alphas):
        threshold = 1e-5

        sv = ((alphas > threshold) * (alphas < self.C)).flatten()
        self.w = np.dot(X[sv].T, alphas[sv] * self.y[sv, np.newaxis])
        self.b = np.mean( self.y[sv, np.newaxis] - self.alphas[sv] * self.y[sv, np.newaxis] * self.K[sv, sv][:, np.newaxis] )

        return sv


if __name__ == "__main__":
    np.random.seed(42)
    K = 2
    D = 2
    N = 50
    X = np.zeros((N * K, D))
    y = np.zeros(N * K) 

    for j in range(K):
        ix = range(N * j, N * (j + 1))
        r = np.linspace(0.0, 1, N) 
        t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2 
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = j

    y[y == 0] -= 1

    svm = SVM(kernel=gaussian)
    svm.fit(X, y)
    y_pred = svm.predict(X)
    plot_contour(X, y, svm)

    print(f"Accuracy: {sum(y==y_pred)/y.shape[0]}")
