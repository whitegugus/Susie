import numpy as np

import cv_lasso
from susie import susie
import time
from cv_lasso import cv_lasso
from stepwise_regression import Stepwise_regression
from sklearn.linear_model import LassoCV

class sim:
    def __init__(self, n, p, nval, rho=0.9, s=5, snr=1., nrep=20, beta_type=2, l=10):
        self.n = n
        self.p = p
        self.nval = nval
        self.rho = rho
        self.s = s
        self.snr = snr
        self.nrep = nrep
        self.betatype = beta_type
        self.l = l

    def sim_xy(self, n, p, nval, rho=0., s=5, beta_type=1, snr=1.):
        x = np.random.randn(n, p)
        xval = np.random.randn(nval, p)

        if rho != 0:
            index = np.ones((p, p)) * range(p)
            sigma = np.power(rho, np.abs(index - index.T))
            eigen, u = np.linalg.eig(sigma)
            sigma_half = u.dot(np.diag(np.sqrt(eigen))).dot(u.T)
            x = x.dot(sigma_half)
            xval = xval.dot(sigma_half)
        else:
            sigma = np.eye(p)

        s = min(s, p)
        beta = np.zeros(p)
        if beta_type == 1:
            beta[(np.around(np.linspace(0, p-1, num=s))).astype('int')] = 1
        elif beta_type == 2:
            beta[range(s)] = 1
        elif beta_type == 3:
            beta[range(s)] = np.linspace(0.5, 10, num=s)
        elif beta_type == 4:
            beta[range(s)] = 1
            beta[s:p] = np.power(0.5, range(1, p-s+1))

        var = np.sum(beta * np.dot(sigma, beta))
        sigma_1 = (var / snr) ** 0.5

        y = np.dot(x, beta) + np.random.randn(n) * sigma_1
        yval = np.dot(xval, beta) + np.random.randn(nval) * sigma_1

        return {'x': x, 'y': y, 'xval': xval, 'yval': yval, 'sigma': sigma, 'sigma_1': sigma_1, 'beta': beta}

    def sim_method(self, method='susie', seed=False):
        if seed:
            np.random.seed(seed)

        summary = np.dtype([('train_err', 'f8'), ('val_err', 'f8'), ('test_err', 'f8'), ('prop', 'f8'), ('risk', 'f8')
                            , ('nzs', 'f8'), ('fp', 'f8'), ('fn', 'f8'), ('F1', 'f8'), ('runtime', 'f8')])

        nul = np.array(tuple(np.full(10, np.nan)), dtype=summary)
        intermediate = np.full(self.nrep, nul, dtype=summary)
        risk_null = np.full(self.nrep, np.nan)
        err_null = np.full(self.nrep, np.nan)
        sigma_1 = np.full(self.nrep, np.nan)

        for i in range(self.nrep):
            xy = self.sim_xy(self.n, self.p, self.nval, self.rho, self.s, self.betatype, self.snr)

            risk_null[i] = np.sum(xy['beta'] * np.dot(xy['sigma'], xy['beta']))
            err_null[i] = risk_null[i] + xy['sigma_1'] ** 2
            sigma_1[i] = xy['sigma_1']
            train_fit = xy['y']
            val_fit = xy['y']
            betahat = xy['beta']
            intercept = 0
            nzs_ind = np.zeros(self.p)
            end = 0
            start = 0

            if method == 'susie':
                fit = susie(xy['x'], xy['y'], l=self.l)
                start = time.time()
                fit.train()
                end = time.time()
                coef = fit.coef()
                cs = fit.get_cs()
                if len(cs['cs_index']):
                    cs = cs['cs_member']
                    l = []
                    for s in cs:
                        l = np.append(l, s)
                    nzs_ind[(l-1).astype('i8')] = 1

                intercept = coef['intercept']
                betahat = coef['b']

                train_fit = fit.predict()
                val_fit = fit.predict(xy['xval'], newx=True)
            elif method == 'lasso':
                fit = cv_lasso()
                start = time.time()
                fit.cross_validation(xy['x'], xy['y'])
                end = time.time()
                intercept = fit.bestintercept
                betahat = fit.bestcoef

                nzs_ind = (betahat != 0)
                train_fit = fit.predict(xy['x'])
                val_fit = fit.predict(xy['xval'])
            elif method == 'step':
                fit = Stepwise_regression(xy['x'], xy['y'])
                start = time.time()
                fit.search()
                end = time.time()
                intercept = fit.coef()['intercept']
                betahat = fit.coef()['betahat']

                nzs_ind = betahat != 0
                train_fit = fit.predict(xy['x'])
                val_fit = fit.predict(xy['xval'])
            elif method == 'Lassocv':
                fit = LassoCV(cv=5)
                start = time.time()
                fit.fit(xy['x'], xy['y'])
                end = time.time()
                intercept = fit.intercept_
                betahat = fit.coef_

                nzs_ind = betahat != 0
                train_fit = fit.predict(xy['x'])
                val_fit = fit.predict(xy['xval'])

            intermediate[i]['runtime'] = end - start
            intermediate[i]['train_err'] = np.mean((xy['y'] - train_fit) ** 2)
            intermediate[i]['val_err'] = np.mean((xy['yval'] - val_fit) ** 2)

            delta = betahat - xy['beta']
            intermediate[i]['risk'] = np.sum(delta * np.dot(xy['sigma'], delta)) + intercept ** 2
            intermediate[i]['test_err'] = intermediate[i]['risk'] + xy['sigma_1'] ** 2
            intermediate[i]['prop'] = 1 - intermediate[i]['test_err'] / err_null[i]
            intermediate[i]['nzs'] = np.sum(nzs_ind)
            tpos = np.sum(nzs_ind * (xy['beta'] != 0))
            intermediate[i]['fp'] = intermediate[i]['nzs'] - tpos
            intermediate[i]['fn'] = np.sum((nzs_ind == 0) * (xy['beta'] != 0))
            intermediate[i]['F1'] = 2 * tpos / (2 * tpos + intermediate[i]['fp'] + intermediate[i]['fn'])

        output = nul
        output['train_err'] = np.mean(intermediate['train_err'])
        output['val_err'] = np.mean(intermediate['val_err'])
        output['risk'] = np.mean(intermediate['risk'] / risk_null)
        output['test_err'] = np.mean(intermediate['test_err'] / (sigma_1 ** 2))
        output['prop'] = np.mean(intermediate['prop'])
        output['nzs'] = np.mean(intermediate['nzs'])
        output['fp'] = np.mean(intermediate['fp'])
        output['fn'] = np.mean(intermediate['fn'])
        output['F1'] = np.mean(intermediate['F1'])
        output['runtime'] = np.mean(intermediate['runtime'])

        std = {'risk': np.std(intermediate['risk'] / risk_null), 'test_err': np.std(intermediate['test_err'] / sigma_1),
               'nzs': np.std(intermediate['nzs'])}
        return output, std


if __name__ == '__main__':
    sim1 = sim(n=500, nval=100, p=100, rho=0.1, s=20, snr=10, beta_type=2,l=20)
    output = sim1.sim_method('susie')
    print(output)
