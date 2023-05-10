import numpy as np


class lasso():
    def __init__(self, x, y, lambda_=1.0, max_iteration=100, fit_intercept=True, tol=0.001):
        self.lambda_ = lambda_
        self.max_iteration = max_iteration
        self.tol = tol

        self.fit_intercept = fit_intercept
        self.intercept = 0
        self.coef= None

        self.xmean = np.mean(x, axis=0)
        self.xscale = np.std(x, axis=0)
        self.x = (x - self.xmean) / self.xscale

        self.y = y
        self.ymean =np.mean(y)

    def soft_thresholding(self, b1, b2):
        if b1 + b2 < 0:
            return b1+b2
        elif b1 - b2 > 0:
            return b1-b2
        else:
            return 0

    def fit(self):
        n = len(self.y)
        p = self.x.shape[1]
        self.coef = np.zeros(p)
        coef_ = self.coef

        if self.fit_intercept:
            self.intercept = self.ymean
            self.y = self.y - self.ymean

        for i in range(self.max_iteration):
            for j in range(p):
                norm = np.sum(self.x[:, j] * self.x[:, j])
                coef_[j] = 0
                res = self.y - np.dot(self.x, coef_)
                b1 = np.sum(self.x[:, j] * res)
                b2 = self.lambda_ * n

                coef_[j] = self.soft_thresholding(b1, b2) / norm

            if np.linalg.norm(coef_ - self.coef, ord=1) < self.tol:
                break

        self.y = self.y + self.intercept
        self.intercept = self.intercept - np.sum(self.xmean * self.coef / self.xscale)
        self.coef = self.coef / self.xscale

    def get_coef(self):
        return {'intercept': self.intercept, 'coef': self.coef}

    def predict(self, x, originx=False):
        if originx:
            return np.dot((self.x + self.xmean) * self.xscale, self.coef) + self.intercept
        return np.dot(x, self.coef) + self.intercept

    def score(self, x, y):
        y_pred = self.predict(x)
        res = np.sum((y - y_pred) ** 2)
        tot = np.sum((y - np.mean(y)) ** 2)
        pve = 1 - res / tot

        return pve


if __name__ == '__main__':
    np.random.seed(0)

    n = 1000
    p = 1000
    beta = np.zeros(p)
    beta[0:4] = 1
    x = np.random.random(size=(n, p))
    x[:, 4] = x[:, 0]
    y = np.dot(x, beta) + np.random.random(size=n)

    fit = lasso(x, y, lambda_=0.02)
    fit.fit()
    print(fit.get_coef())
    print(np.mean(y))





