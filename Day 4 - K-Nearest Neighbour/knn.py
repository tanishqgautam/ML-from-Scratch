import numpy as np

class KNearestNeighbor:
    def __init__(self, X,y,k):
        self.X = X
        self.y = y
        self.k = k
        self.distances = np.zeros((X.shape[0], X.shape[0]))
        self.ypred = np.zeros(self.distances.shape[0])

    def l2_distance(self):

        return np.sqrt( 
                        np.sum(self.X ** 2, axis=1) - 
                        2 * np.dot(self.X, self.X.T) + 
                        np.sum(self.X*self.X, axis=1)[None].T 
                      )

    def main(self):
        distances = self.l2_distance()

        for i in range(distances.shape[0]):
            y_values = np.argsort(distances[i, :])
            k_closest_classes = self.y[y_values[: self.k]].astype(int)
            self.ypred[i] = np.argmax(np.bincount(k_closest_classes))

        return self.ypred


if __name__ == "__main__":

    X = np.array([[1, 1], [3, 1], [1, 4], [2, 4], [3, 3], [5, 1]])
    y = np.array([0., 0., 0., 1., 1., 1.])

    KNN = KNearestNeighbor(X, y,k=2)
    y_pred = KNN.main()

    print("Expected y = ", y)
    print("Predicted y = ", y_pred)
    print(f"Accuracy: {sum(y_pred == y) / y.shape[0]}")
