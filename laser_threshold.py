import numpy as np
import pqrc
import matplotlib.pyplot as plt
import time
from joblib import Parallel, delayed
import multiprocessing as mp

# wrapper for g2 calculation
def calc_g2(pp):
    gamma_r = 4 * g**2 / (pp + kappa + gamma_d + gamma_a)
    rates = (gamma_r, kappa, gamma_a, pp)
    t_array, na_out_array, na_in_array, ne_array, phi_array, dtau_array = pqrc.laser.laser_timeseries(na_init, ne_init, n0, rates, alpha, num_points)
    g2 = pqrc.laser.calc_g2_0(na_in_array, t_array, start_ind=10000)
    return g2

start = time.time()

# paramters taken from Moerk Fig.2
n0 = 5
g = .1
kappa = 0.04 
gamma_a = 0.012
gamma_d = 1.
alpha = 0

num_points = int(2e+6)
num_cores = mp.cpu_count()

# inital conditions
na_init = 0
ne_init = 0

# calculations
pump_array = np.logspace(-3, 2, num=24) / n0
g2_array = Parallel(n_jobs=num_cores)(delayed(calc_g2)(pp) for pp in pump_array)
print(time.time() - start)

# plotting
fig, ax = plt.subplots()
ax.plot(pump_array * n0, g2_array, linestyle='-', marker='o', color='black')

ax.set_xscale('log')
ax.set_xlabel(r'$n_0 P$ [ps$^{-1}$]')
ax.set_ylabel(r'$g^{(2)}(0)$')

fig.savefig('plots/g2_pump.pdf')
fig.show()