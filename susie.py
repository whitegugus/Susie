import numpy as np
from scipy.stats import norm
import math as ma


class susie():
    def __init__(self, input_data, target_data, l=10, scaled_prior_variance=0.2, iteration=1000, tol=0.001):
        self.xmean = np.mean(input_data, axis=0)
        self.xscale = np.std(input_data, axis=0)
        self.input_data = (input_data - self.xmean) / self.xscale
        self.xtx = np.sum(self.input_data * self.input_data, axis=0)

        self.ymean = np.mean(target_data)
        self.yscale = np.std(target_data)
        self.target_data = (target_data - self.ymean) / self.yscale

        self.length = len(target_data)
        self.p = input_data.shape[1]
        self.l = l
        self.max_iteration = iteration
        self.iterations = 0
        self.tol = tol

        self.prior = np.ones(self.p) / self.p
        self.b_mean = np.zeros((self.l, self.p))
        self.b_mean2 = np.ones((self.l, self.p))
        self.gamma = np.ones((self.l, self.p)) / self.p

        self.xr = np.zeros(self.length)
        self.kl = np.full(self.l, np.nan)
        self.lbf = np.full(self.l, np.nan)
        self.lbf_variable = np.full((self.l, self.p), np.nan)

        self.elbo = np.full(iteration+1, np.nan)
        self.elbo[0] = float('-inf')

        self.sigma_1 = np.var(target_data)
        self.sigma_0 = np.ones(l) * self.sigma_1 * scaled_prior_variance

        pass

    def ser(self, y, sigma_0):
        xty = np.dot(self.input_data.T, y)
        betahat = (1 / self.xtx) * xty
        shat2 = self.sigma_1 / self.xtx

        lbf = norm.logpdf(betahat, 0, np.sqrt(sigma_0 + shat2)) - norm.logpdf(betahat, 0, np.sqrt(shat2))

        lbf[np.isinf(shat2)] = 0
        maxlbf = np.max(lbf)

        w = np.exp(lbf - maxlbf)
        w_weighted = w * self.prior
        weighted_sum = np.sum(w_weighted)
        gamma = w_weighted / weighted_sum
        post_var = 1 / (1 / sigma_0 + self.xtx / self.sigma_1)
        post_mean = (1 / self.sigma_1) * post_var * xty
        post_mean2 = post_var + post_mean * post_mean

        lbf_model = maxlbf + ma.log(weighted_sum)
        loglike = lbf_model + np.sum(norm.logpdf(y, 0, ma.sqrt(self.sigma_1)))

        return {'gamma': gamma, 'post_mean': post_mean, 'post_mean2': post_mean2, 'lbf': lbf,
                'lbf_model': lbf_model, 'sigma_0': sigma_0, 'loglike': loglike}

    def update(self):
        for i in range(0, self.l):
            self.xr = self.xr - np.dot(self.input_data, (self.gamma[i, :] * self.b_mean[i, :]))
            R = self.target_data - self.xr

            res = self.ser(R, self.sigma_0[i])

            self.b_mean[i, :] = res['post_mean']
            self.gamma[i, :] = res['gamma']
            self.b_mean2[i, :] = res['post_mean2']
            self.sigma_0[i] = res['sigma_0']
            self.lbf[i] = res['lbf_model']
            self.lbf_variable[i, :] = res['lbf']

            self.kl[i] = -res['loglike'] - 0.5 * self.length * ma.log(2 * 3.14 * self.sigma_1) - 0.5 / self.sigma_1 \
            * (np.sum(R * R) - 2 * np.sum(R * np.dot(self.input_data, res['gamma'] * res['post_mean']))) + \
                         np.sum(self.xtx * (res['gamma'] * res['post_mean2']))

            self.xr = self.xr + np.dot(self.input_data, self.gamma[i, :] * self.b_mean[i, :])
        pass

    def get_elbo(self):
        return self.get_eloglike() - np.sum(self.kl)

    def get_eloglike(self):
        return -(self.length/2) * ma.log(2 * 3.14 * self.sigma_1) - (1 / (2 * self.sigma_1)) * self.get_er2()

    def get_er2(self):
        xr_total = np.dot(self.input_data, (self.gamma * self.b_mean).T)
        postb2 = self.gamma * self.b_mean2
        return np.sum((self.target_data - self.xr) ** 2) - np.sum(xr_total * xr_total) + np.sum(self.xtx * postb2)

    def train(self):
        for i in range(0, self.max_iteration):
            self.iterations = i+1
            self.update()
            self.sigma_1 = np.var(self.target_data - self.xr)
            self.elbo[i+1] = self.get_elbo()
            if self.tol > (self.elbo[i + 1] - self.elbo[i]) > 0:
                break

        self.elbo = self.elbo[1:self.iterations + 1]
        pass

    def coef(self):
        eb = np.sum(self.gamma * self.b_mean, axis=0)
        b = self.yscale * eb / self.xscale
        intercept = self.ymean - np.sum(self.xmean * b)
        return {'intercept': intercept, 'b': b}

    def predict(self, x=None, newx=False):
        if not newx:
            return self.xr * self.yscale + self.ymean
        else:
            coef = self.coef()
            intercept = coef['intercept']
            b = coef['b']
            return intercept + np.dot(x, b)

    def get_pip(self):
        return 1 - np.exp(np.sum(np.log(1 - self.gamma), axis=0))

    def get_cs(self, coverage=0.95, n_purity=100, min_abs_cor=0.3):
        cs_index = list()
        cs_member = list()
        cs_coverage = list()
        cs_min_abs_cor = list()
        x_cor = np.corrcoef(self.input_data.T)

        for i in range(0, self.l):
            order = np.argsort(-self.gamma[i, :])
            ordered_gamma = self.gamma[i, order]
            effect = 0
            n = 0
            for j in range(0, self.p):
                effect += ordered_gamma[j]
                if effect > coverage:
                    n = j
                    break
            index = order[0:n+1]
            corM = np.absolute(x_cor[np.ix_(index, index)])
            mcor = np.min(corM)
            if mcor < min_abs_cor:
                continue
            cs_index.append(i+1)
            cs_member.append(index+1)
            cs_coverage.append(effect)
            cs_min_abs_cor.append(mcor)

        return {'cs_index':cs_index, 'cs_member':cs_member, 'cs_coverage':cs_coverage, 'cs_min_abs_cor':cs_min_abs_cor}


if __name__ == '__main__':
    n=500
    p=100
    rho=0.1
    s=20
    x = np.random.randn(n, p)
    snr= 2

    beta_type = 2

    index = np.ones((p, p)) * range(p)
    sigma = np.power(rho, np.abs(index - index.T))
    eigen, u = np.linalg.eig(sigma)
    sigma_half = u.dot(np.diag(np.sqrt(eigen))).dot(u.T)
    x = x.dot(sigma_half)

    s = min(s, p)
    beta = np.zeros(p)
    if beta_type == 1:
        beta[np.around(np.linspace(0, p - 1, num=s))] = 1
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

    print(np.shape(x), np.shape(y))

    fit = susie(x, y, iteration=1000, l=20)
    print(fit.length)
    fit.train()
    print(fit.coef())
    print(fit.get_cs())







