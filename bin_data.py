import numpy as np
import matplotlib.pyplot as plt

import laser


def get_time_series_bin_data(pp: float, bin_size: int):

    n0 = 5
    g = .1
    kappa = 0.04
    gamma_a = 0.012
    gamma_d = 1.
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

    # bin outcoupled photons
    t_binned, a_binned = laser.bin_laser_timeseries(t_array, na_out_array, bin_size)
    g2 = laser.calc_g2_0(na_in_array, t_array, start_ind=1000)

    print('\nt_binned')
    print(t_binned)

    print('\na_binned')
    print(a_binned)

    return t_binned, a_binned, g2


if __name__ == '__main__':

    pp = 0.6
    bin_size = 2

    t_binned, a_binned, g2 = get_time_series_bin_data(pp, bin_size)

    print('\ng2: {:.3f}'.format(g2))

    print(t_binned.shape)
    print(a_binned.shape)

    fig, ax = plt.subplots()

    ax.plot(t_binned, a_binned)
    plt.show()
