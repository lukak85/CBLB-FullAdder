import numpy as np
from scipy.integrate import ode
import matplotlib.pyplot as plt

from models import *
from parameters import *

params = (delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x, gamma_x, theta_x, r_X)
sums = []
x_axis = []
y_axis = []
z_axis = []

for ia in range(0, 2):

    x_axis.append(ia)
    y_axis.insert(0, ia)

    for ib in range(0, 2):

        sums.insert(0, [])
        y_axis.append(ib)
        z_axis.insert(0, ib)

        for ic in range(0,2):

            t_end = 400
            N = t_end
            rho_x = 0
            Y0 = np.zeros(56)

            Y0[0] = ia * 10 # A
            Y0[1] = ib * 10 # B
            Y0[2] = ic * 10 # C

            Y0[3] = 1
            Y0[5] = 1
            Y0[6] = 1
            Y0[7] = 1
            Y0[8] = 1
            Y0[10] = 1
            Y0[11] = 1
            Y0[12] = 1
            Y0[13] = 1

            for i in range(15, 54):
                if i % 2 == 1:
                    Y0[i] = 1
                    
            params = (delta_L, gamma_L_X, n_y, theta_L_X, eta_x, omega_x, m_x, delta_x, rho_x, gamma_x, theta_x, r_X)



            # initialization
            T = np.linspace(0, t_end, N)

            t1 = t_end
            dt = t_end / N
            T = np.arange(0, t1 + dt, dt)
            Y = np.zeros([1 + N, 56])
            Y[0, :] = Y0



            # simulation
            r = ode(FA_model).set_integrator('zvode', method='bdf')
            r.set_initial_value(Y0, T[0]).set_f_params(params)

            i = 1
            while r.successful() and r.t < t1:
                Y[i, :] = r.integrate(r.t + dt)
                i += 1

            
            S = Y[:, 51]
            Cout = Y[:, 55]

            # print(S)
            # print(Cout)
            # print()

            S_RES = 1 if S[-1] > 1 else 0
            S_CARR = 1 if Cout[-1] > 1 else 0

            finalResult = S_CARR * 2 + S_RES

            print(finalResult)

            # plt.plot(S, label="S")
            # plt.plot(Cout, label="Cout")
            # plt.plot(Y[:,0], label="A")
            # plt.plot(Y[:,1], label="B")
            # plt.plot(Y[:,2], label="C")
            # plt.legend()
            # plt.show()

            sums[0].append(finalResult)

print(sums)

Bhm = ["0", "1"]
Chm = ["0", "1"]

Sum0 = np.array([sums[3], sums[2]])

fig, ax = plt.subplots()
im = ax.imshow(Sum0)

ax.set_xticks(np.arange(len(Bhm)), labels=Bhm)
ax.set_yticks(np.arange(len(Chm)), labels=Chm)

for i in range(len(Bhm)):
    for j in range(len(Chm)):
        text = ax.text(j, i, Sum0[i, j],
                       ha="center", va="center", color="w")

ax.set_title("A = 0")
fig.tight_layout()
cbar = ax.figure.colorbar(im, ax=ax)
plt.show()

Sum0 = np.array([sums[1], sums[0]])

fig, ax = plt.subplots()
im = ax.imshow(Sum0)

ax.set_xticks(np.arange(len(Bhm)), labels=Bhm)
ax.set_yticks(np.arange(len(Chm)), labels=Chm)

for i in range(len(Bhm)):
    for j in range(len(Chm)):
        text = ax.text(j, i, Sum0[i, j],
                       ha="center", va="center", color="w")

ax.set_title("A = 1")
fig.tight_layout()
cbar = ax.figure.colorbar(im, ax=ax)
plt.show()