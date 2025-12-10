import numpy as np
import matplotlib.pyplot as plt

import laser

# Parameters taken from Moerk Fig. 2
n0 = 5
g = .1
kappa = 0.04
gamma_a = 0.012
gamma_d = 1.
# pp = .6
# pp = .01
pp = 10.
alpha = 0

# resulting rates
gamma_r = 4 * g**2 / (pp + kappa + gamma_d + gamma_a)
rates = (gamma_r, kappa, gamma_a, pp)

# initial conditions
na_init = 0
ne_init = 0

# calculate timeseries
num_points = int(1e+5)
t_array, na_out_array, na_in_array, ne_array, phi_array, dtau_array = (
    laser.laser_timeseries(na_init, ne_init, n0, rates, alpha, num_points)
)

if pp == .6:
    bin_size = 2.
    tmin = 5e+3
    tmax = 5e+3 + 20
    twin = bin_size * 20


if pp == .01:
    bin_size = 50
    tmin = 5e+4
    tmax = 5e+4 + 20
    twin = bin_size * 50

if pp == 10.:
    bin_size = 50
    tmin = 5e+4
    tmax = 5e+4 + 20
    twin = bin_size * 50

# bin outcoupled photons
t_binned, a_binned = laser.bin_laser_timeseries(t_array, na_out_array, bin_size)
g2 = laser.calc_g2_0(na_in_array, t_array, start_ind=1000)

print('t_max =', max(t_array), 'ps')
print('g^2(0) =', g2)

fig, ax = plt.subplots(2,2, figsize=(8,6))

ax[0,0].plot(t_array-tmin, na_out_array)
ax[0,1].plot(t_array-tmin, na_in_array)
ax[1,0].plot(t_array-tmin, ne_array)
ax[1,1].plot(t_binned-tmin, a_binned)

ax[0,0].set_xlabel(r'$t$ [ps]')
ax[0,0].set_ylabel(r'$n_\mathrm{out}(t)$')
ax[0,0].set_xlim((0, twin))

ax[0,1].set_xlabel(r'$t$ [ps]')
ax[0,1].set_ylabel(r'$n_\mathrm{a}(t)$')
ax[0,1].set_xlim((0, twin))

ax[1,0].set_xlabel(r'$t$ [ps]')
ax[1,0].set_ylabel(r'$n_\mathrm{e}(t)$')
ax[1,0].set_xlim((0, twin))

ax[1,1].set_xlabel(r'$t$ [ps]')
ax[1,1].set_ylabel(r'$\bar{n}_\mathrm{out}(t)$')
ax[1,1].set_xlim((0, twin))


fig.suptitle(r'$p={:.2f}$\,ps$^{{-1}}$, $g^{{(2)}}(0) = {:.2f}$'.format(pp, g2))
fig.show()
fig.savefig('plots/timeseries_p_{}.pdf'.format(pp))

