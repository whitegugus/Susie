import numpy as np
import matplotlib.pyplot as plt

z = np.load('data_new4.npz')
data = z['data']
rho = z['rho']
betatype = z['betatype']
method = z['method']

snr = z['snr']

color = ['r', 'g', 'b']
x = snr

fig0, axes0 = plt.subplots(len(betatype), len(rho), figsize=(12, 9), dpi=100, sharex='all', sharey='all')

for i, row in enumerate(axes0):
    for j, col in enumerate(row):
        for k in range(len(method)):
            axes0[i, j].plot(np.log10(x), data[k, j, i, :, 0], color=color[k], label=method[k])

            axes0[i, j].grid()
            if i == 0:
                axes0[i, j].set_title('Correlation = '+str(rho[j]))
            if j == 0:
                axes0[i, j].set_ylabel('Betatype '+str(betatype[i]))
            if (i == 0)*(j == len(method)-1):
                axes0[i, j].legend()

fig0.text(0.5, 0.05, 'Signal-to-noise ratio ', ha='center')
fig0.text(0.05, 0.5, 'Relative risk (to null model)', va='center', rotation='vertical')

fig1, axes1 = plt.subplots(len(betatype), len(rho), figsize=(12, 9), dpi=100, sharex='all', sharey='all')

for i, row in enumerate(axes1):
    for j, col in enumerate(row):
        for k in range(len(method)):
            axes1[i, j].plot(np.log10(x), data[k, j, i, :, 1], color=color[k], label=method[k])

            axes1[i, j].grid()
            if i == 0:
                axes1[i, j].set_title('Correlation = '+str(rho[j]))
            if j == 0:
                axes1[i, j].set_ylabel('Betatype '+str(betatype[i]))
            if (i == 0)*(j == len(rho)-1):
                axes1[i, j].legend()


fig1.text(0.5, 0.05, 'Signal-to-noise ratio ', ha='center')
fig1.text(0.05, 0.5, 'Relative risk (to null model)', va='center', rotation='vertical')

fig2, axes2 = plt.subplots(len(betatype), len(rho), figsize=(12, 9), dpi=100, sharex='all', sharey='all')

for i, row in enumerate(axes2):
    for j, col in enumerate(row):
        for k in range(len(method)):
            axes2[i, j].plot(np.log10(x), data[k, j, i, :, 2], color=color[k], label=method[k])

            axes2[i, j].grid()
            if i == 0:
                axes2[i, j].set_title('Correlation = '+str(rho[j]))
            if j == 0:
                axes2[i, j].set_ylabel('Betatype '+str(betatype[i]))
            if (i == 0)*(j == len(method)-1):
                axes2[i, j].legend()


fig2.text(0.5, 0.05, 'Signal-to-noise ratio ', ha='center')
fig2.text(0.05, 0.5, 'Relative risk (to null model)', va='center', rotation='vertical')

fig3, axes3 = plt.subplots(len(betatype), len(rho), figsize=(12, 9), dpi=100, sharex='all', sharey='all')

for i, row in enumerate(axes1):
    for j, col in enumerate(row):
        for k in range(len(method)):
            axes3[i, j].plot(np.log10(x), data[k, j, i, :, 3], color=color[k], label=method[k])

            axes3[i, j].grid()
            if i == 0:
                axes3[i, j].set_title('Correlation = '+str(rho[j]))
            if j == 0:
                axes3[i, j].set_ylabel('Betatype '+str(betatype[i]))
            if (i == 0)*(j == len(method)-1):
                axes3[i, j].legend()


fig3.text(0.5, 0.05, 'Signal-to-noise ratio ', ha='center')
fig3.text(0.05, 0.5, 'Relative risk (to null model)', va='center', rotation='vertical')
plt.show()