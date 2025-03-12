import numpy as np

class LogisticRegressor:
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y, learning_rate=0.01, num_iterations=1000, penalty=None, l1_ratio=0.5, C=1.0, verbose=False, print_every=100):
        m, n = X.shape
        self.weights = np.zeros(n)
        self.bias = 0

        for i in range(num_iterations):
            y_hat = self.predict_proba(X)
            loss = self.log_likelihood(y, y_hat)

            if i % print_every == 0 and verbose:
                print(f"Iteration {i}: Loss {loss}")

            dw = (1 / m) * np.dot(X.T, (y_hat - y))
            db = (1 / m) * np.sum(y_hat - y)

            if penalty == "lasso":
                dw = self.lasso_regularization(dw, m, C)
            elif penalty == "ridge":
                dw = self.ridge_regularization(dw, m, C)
            elif penalty == "elasticnet":
                dw = self.elasticnet_regularization(dw, m, C, l1_ratio)

            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)

    def lasso_regularization(self, dw, m, C):
        lasso_gradient = (C / m) * np.sign(self.weights)
        return dw + lasso_gradient

    def ridge_regularization(self, dw, m, C):
        ridge_gradient = (C / m) * self.weights
        return dw + ridge_gradient

    def elasticnet_regularization(self, dw, m, C, l1_ratio):
        lasso_grad = self.lasso_regularization(dw, m, C) * l1_ratio
        ridge_grad = self.ridge_regularization(dw, m, C) * (1 - l1_ratio)
        return lasso_grad + ridge_grad

    @staticmethod
    def log_likelihood(y, y_hat):
        m = y.shape[0]
        loss = -(1 / m) * np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
        return loss

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
