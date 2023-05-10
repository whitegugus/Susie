import numpy as np
import sim
import math as ma

n = 500
nval = 100
p = 100
s = 50
rho = [0.1, 0.4, 0.7]
snr = np.logspace(ma.log10(0.05), ma.log10(10), num=12, base=10)
betatype = [1, 2, 3, 4]
method = ['susie', 'Lassocv', 'step']
parameters = ['risk', 'test_err', 'nzs', 'prop']
seed = 0
nrep = 20
L = 50


data = np.zeros((len(method), len(rho), len(betatype), len(snr), len(parameters)))


for j in range(len(rho)):
    for k in range(len(betatype)):
        for l in range(len(snr)):
            print(l)
            fit = sim.sim(n=n, nval=nval, p=p, s=s, rho=rho[j], snr=snr[l], beta_type=betatype[k], nrep=nrep, l=L)
            for i in range(len(method)):
                output, std = fit.sim_method(method[i], seed=seed)
                for m in range(len(parameters)):
                    data[i, j, k, l, m] = output[parameters[m]]

np.savez('data_new4.npz', data=data, snr=snr, rho=rho, betatype=betatype, method=method, parameters=parameters)
