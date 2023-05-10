import numpy as np


class Stepwise_regression:
    def __init__(self, x, y):
        self.x_mean = np.mean(x, axis=0)
        self.x_scale = np.std(x, axis=0)
        self.x_c = (x - self.x_mean) / self.x_scale

        self.y_mean = np.mean(y)
        self.y_c = y - self.y_mean

        self.betahat = np.zeros(np.shape(x)[1])
        self.intercept = self.y_mean

        self.A = np.dot(np.c_[self.x_c, self.y_c].T, np.c_[self.x_c, self.y_c])

    def eliminate_trans(self, m, index):
        n = len(m)
        m1 = m.copy()
        center = m[index, index]
        m1[index, index] = 1 / center

        for j in range(n):
            if j != index:
                m1[index, j] = m[index, j] / center
                m1[j, index] = - m[j, index] / center

        for i in range(n):
            if i != index:
                for j in range(n):
                    if j != index:
                        m1[i, j] = m[i, j] - m[i, index] * m[index, j] / center
        return m1

    def compute_aic(self, rss, n, q):
        return n * np.log(rss) + q

    def search(self):
        n = len(self.y_c)
        p = np.shape(self.x_c)[1]

        in_effect = np.zeros(p)
        out_effect = np.ones(p)

        aic_path = [self.compute_aic(self.A[p, p], n, 0)]
        operate_path = []
        best_aic = aic_path[0]
        q = 0

        while 1:
            p_value = (self.A[0:p, p] ** 2) / np.diag(self.A)[0:p]

            aic_add = best_aic + 1
            aic_del = best_aic + 1
            ind_add = p + 1
            ind_del = p + 1

            if q <= p-1:
                ind_add = np.argmax(out_effect * p_value)
                rss_add = self.A[p, p] - p_value[ind_add]
                aic_add = self.compute_aic(rss_add, n, q + 1)

            if q >= 1:
                nz_ind = np.where(in_effect > 0)[0]
                ind_del = nz_ind[np.argmin((in_effect * p_value)[nz_ind])]
                rss_del = self.A[p, p] + p_value[ind_del]
                aic_del = self.compute_aic(rss_del, n, q - 1)

            aic_next = min(best_aic, aic_add, aic_del)

            if aic_next == best_aic:
                break
            elif aic_next == aic_add:
                best_aic = aic_next
                q += 1
                in_effect[ind_add] = 1
                out_effect[ind_add] = 0
                operate_path.append(ind_add)
                aic_path.append(best_aic)
                self.A = self.eliminate_trans(self.A, ind_add)
            else:
                best_aic = aic_next
                q -= 1
                out_effect[ind_del] = 1
                in_effect[ind_del] = 0
                operate_path.append(-ind_del)
                aic_path.append(best_aic)

                self.A = self.eliminate_trans(self.A, ind_del)

        self.betahat[in_effect == 1] = self.A[0:p, p][in_effect == 1]
        self.betahat = self.betahat / self.x_scale
        self.intercept = self.intercept - np.sum(self.betahat * self.x_mean)

        return {'selected_effect': in_effect, 'operate_path': operate_path, 'AIC_path': aic_path}

    def coef(self):
        return {'intercept': self.intercept, 'betahat': self.betahat}

    def predict(self, x, newx=True):
        if newx:
            return self.intercept + np.dot(x, self.betahat)
        else:
            return self.intercept + np.dot((self.x_c + self.x_mean) * self.x_scale, self.betahat)


if __name__ == '__main__':
    np.random.seed(0)

    n=100
    p=200
    rho=0.5
    s=5
    x = np.random.randn(n, p)
    snr=10

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
        beta[s:p] = np.power(0.5, range(1, p - s + 1))

    var = np.sum(beta * np.dot(sigma, beta))
    sigma_1 = (var / snr) ** 0.5

    y = np.dot(x, beta) + np.random.randn(n) * sigma_1

    fit = Stepwise_regression(x, y)
    path = fit.search()
    print(path)
    print(fit.coef())












