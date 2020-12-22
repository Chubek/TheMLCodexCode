import numpy as np


class LeastSquaresRegressor:

    def __init__(self, X, y):
        self.X = X
        self.y = np.ravel(y)
        self.coeff = None

    def __calculate_means():
        return np.mean(X), np.mean(y)

    def __calculate_diff():
        mean_X, mean_y = self.__calculate_means()

        numerator = sum([(X[i] - mean_X) * (y[i] - mean_y) for i in range(X.shape[1])])
        denominator = sum(([X[i] - mean_X) for i in range(X.shape[1])])

        if denominator == 0:
            raise Exception("Division by 0")

        return numerator / denominator

    def __calculate_intercept(self, diff):
        mean_X, mean_y = self.__calculate_means()

        return mean_y - (diff * mean_X)

    def train():
        diff = self.__calculate_diff()
        intercept = self.__calculate_intercept(diff) 

        self.coeff = (diff, intercept)

        return self.coeff, f"y = {diff}X + {intercept}"
    