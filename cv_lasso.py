import numpy as np
from lasso import lasso
from sklearn.model_selection import KFold


class cv_lasso:
    def __init__(self, max_iteration=1000, fit_intercept=True, tol=0.001, kfold=5, log_alpha=(-10, 2), nalpha=100):
        self.max_iteration = max_iteration
        self.fit_intercept = fit_intercept
        self.tol =tol

        self.kfold =kfold

        self.scores = []
        self.coef = []
        self.intercept = []

        self.alphas = np.logspace(log_alpha[0], log_alpha[1], nalpha)
        self.bestalpha = 0
        self.bestcoef = []
        self.bestintercept = 0

    def cross_validation(self, x, y):
        kf = KFold(n_splits=self.kfold, shuffle=True, random_state=42)
        p = np.shape(x)[1]
        self.coef = np.zeros((len(self.alphas), p))
        i = 0

        for alpha in self.alphas:
            fold_scores = []
            fold_coef = np.zeros((self.kfold, p))
            fold_intercept = []
            j = 0

            for train_idx, test_idx in kf.split(x):
                x_train, y_train = x[train_idx], y[train_idx]
                x_test, y_test = x[test_idx], y[test_idx]
                model = lasso(x_train, y_train, lambda_=alpha, max_iteration=self.max_iteration,
                              fit_intercept=self.fit_intercept, tol=self.tol)
                model.fit()
                fold_scores.append(model.score(x_test, y_test))
                fold_coef[j, :] = model.coef
                fold_intercept.append(model.intercept)
                j = j + 1

            self.scores.append(np.mean(fold_scores))
            self.coef[i, :] = np.mean(fold_coef, axis=0)
            self.intercept.append(np.mean(fold_intercept))
            i = i + 1

        index = np.argmax(self.scores)
        self.bestalpha = self.alphas[index]
        self.bestcoef = self.coef[index]
        self.bestintercept = self.intercept[index]

    def predict(self, x):
        return self.bestintercept + np.dot(x, self.bestcoef)


if __name__ == '__main__':


    n=100
    p=200
    rho=0.5
    s=5
    x = np.random.randn(n, p)
    snr=0.1

    beta_type = 2

    index = np.ones((p, p)) * range(p)
    sigma = np.power(rho, np.abs(index - index.T))
    eigen, u = np.linalg.eig(sigma)
    sigma_half = u.dot(np.diag(np.sqrt(eigen))).dot(u.T)
    x = x.dot(sigma_half)

    s = min(s, p)
    beta = np.zeros(p)
    if beta_type == 1:
        beta[(np.around(np.linspace(0, p - 1, num=s))).astype('int')] = 1
    elif beta_type == 2:
        beta[range(s)] = 1
    elif beta_type == 3:
        beta[range(s)] = np.linspace(0.5, 10, num=s)
    elif beta_type == 4:
        beta[range(s)] = 1
        beta[s:(p - 1)] = np.power(0.5, range(1, p - s + 1))

    var = np.sum(beta * np.dot(sigma, beta))
    sigma_1 = (var / snr) ** 0.5

    y = np.dot(x, beta) + np.random.randn(n) * sigma_1

    cv = cv_lasso(max_iteration=1000)
    cv.cross_validation(x, y)
    print(cv.bestalpha)
    print(cv.bestcoef)


