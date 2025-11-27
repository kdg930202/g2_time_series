import numpy as np

def calc_event_rates(
    na: int,
    ne: int,
    n0: int,
    gamma_r: float,
    kappa: float,
    gamma_a: float,
    pp: float
) -> np.ndarray[float]:
    r"""
    Calculate event rates :math:`a_\mu`.

    Args:
       na: Number of photons in cavity :math:`n_\mathrm{a}`.
       ne: Number of excited emitters :math:`n_\mathrm{e}`.
       n0: Total emitter number :math:`n_0`.
       gamma_r: Effective light-matter interaction rate :math:`\gamma_\mathrm{r}`.
       kappa: Cavity decay/outcoupling rate :math:`\kappa`.
       gamma_a: Background (non-radiative emitter) decay rate :math:`\gamma_\mathrm{A}`.
       pp: Pump rate :math:`P`.

    Returns:
        Array of event rates :math:`[a_\mathrm{st}, a_\mathrm{sp}, a_\mathrm{ab}, a_\mathrm{c}, a_\mathrm{p}, a_\mathrm{bg}]`.
    """

    a_st = gamma_r * ne * na
    a_sp = gamma_r * ne
    a_ab = gamma_r * (n0 - ne) * na
    a_c  = kappa * na
    a_p  = pp * (n0 - ne)
    a_bg = gamma_a * ne

    return np.array([a_st, a_sp, a_ab, a_c, a_p, a_bg])

def gen_time_increments(event_rates: np.ndarray[float]) -> np.ndarray:
    r"""
    Calculate time increments :math:`\tau_\mu` from event rates :math:`a_\mu` by sampling from

    .. math::
        p(\tau_\mu) = a_\mu \mathrm{e}^{-a_\mu \tau_\mu}

    Args:
       event_rates: Array of event rates :math:`[a_\mathrm{st}, a_\mathrm{sp}, a_\mathrm{ab}, a_\mathrm{c}, a_\mathrm{p}, a_\mathrm{bg}]`.

    Returns:
        Array of time increments :math:`[\tau_\mathrm{st}, \tau_\mathrm{sp}, \tau_\mathrm{ab}, \tau_\mathrm{c}, \tau_\mathrm{p}, \tau_\mathrm{bg}]`.
    """

    tau_mu_array = np.empty_like(event_rates)
    for mu, aa in enumerate(event_rates):
        if aa == 0:
            tau_mu_array[mu] = 1e+10
        else:
            tau_mu_array[mu] = np.random.exponential(1/aa)

    return tau_mu_array

def calc_phase_change(na: int, phi: float) -> float:
    r"""
    Calculate phase shift $\Delta \phi$ using

    .. math::
        \Delta \phi = \mathrm{arg} \left(\sqrt{n_\mathrm{a}\mathrm{e}^{i\phi} + \mathrm{i\theta} }  \right) - \phi,

    where :math:`\theta` is a random number uniformly sampled from :math:`[-\pi, \pi]`.

    Args:
        na: Current number of photons in cavity :math:`n_\mathrm{a}`.
        phi: Current phase :math:`\phi`.

    Returns:
        Change in phase :math:`\Delta \phi`.
    """

    theta = 2 * np.pi * np.random.rand(1)[0]
    current_efield = np.sqrt(na) * np.exp(1j * phi)
    new_efield = np.sqrt(na) * np.exp(1j * phi) + np.exp(1j * theta)
    return np.angle(new_efield - current_efield)

def update_occupations(mu: int, na: int, phi: float) -> tuple:
    r"""
    Calculate the change in occupations and phase for a given event.

    Args:
        mu: Event index :math:`\mu_0`.
        na: Current number of photons in cavity :math:`n_\mathrm{a}`.
        phi: Current phase :math:`\phi`.

    Returns:
        Changes in :math:(`n_\mathrm{a}, n_\mathrm{e}, n_\mathrm{out}, \phi`).
    """
    if mu == 0:
        return (1, -1, 0, 0)
    elif mu == 1:
        dphi = calc_phase_change(na, phi)
        return (1, -1, 0, dphi)
    elif mu == 2:
        return (-1, 1, 0, 0)
    elif mu == 3:
        return (-1, 0, 1, 0)
    elif mu == 4:
        return (0, 1, 0, 0)
    elif mu == 5:
        return (0, -1, 0, 0)

def laser_timeseries(
    na_init: int,
    ne_init: int,
    n0: int,
    rates: np.ndarray[float],
    alpha: float,
    num_points: int
) -> tuple[np.ndarray]:
    r"""
    Calculate the timeseries of the laser emission using the statistical approach.

    Args:
        na_init: Initial photon number :math:`n_\mathrm{a}(0)`.
        ne_init: Initial excited emitter number :math:`n_\mathrm{e}(0)`.
        n0: Total number of emitters :math:`n_0`.
        rates: Array of rates :math:`[\gamma_\mathrm{r}, \kappa, \gamma_\mathrm{A}, P]`.
        alpha: Linewidth enhancement factor :math:`\alpha`.
        num_points: Number of time steps to simulate.

    Returns:
        Array of times :math:`t_i` where an event occurs.
        Array of outcoupled photon number :math:`n_\mathrm{out}(t_i)`.
        Array of photon number in cavity :math:`n_mathrm{a}(t_i)`.
        Array of excited emitter number :math:`n_\mathrm{e}(t_i)`.
        Array of phase :math:`\phi(t_i)`.
        List of cavity emission times :math:`\Delta \tau`.
    """
    # initial values
    na = na_init
    ne = ne_init
    phi = 0
    t_tot = 0.
    delta_tau = 0

    # arrays to save data
    t_array = np.empty(num_points)
    na_out_array = np.empty(num_points, dtype=int)
    na_in_array = np.empty(num_points, dtype=int)
    ne_array = np.empty(num_points, dtype=int)
    phi_array = np.empty(num_points)
    dtau_list = []

    # first values in array
    na_in_array[0] = na
    na_out_array[0] = 0
    ne_array[0] = ne
    t_array[0] = t_tot
    phi_array[0] = phi

    gamma_r = rates[0]


    for ii in range(1, num_points):

        # generate time increments and pick shortest
        event_rates = calc_event_rates(na, ne, n0, *rates)
        tau_mu_array = gen_time_increments(event_rates)
        mu_0 = np.argmin(tau_mu_array)

        # update time since last event
        delta_tau += na * tau_mu_array[mu_0]
        if mu_0 == 1:
            delta_tau = 0
        if mu_0 == 3:
            dtau_list.append(delta_tau)
            delta_tau = 0

        # phase diffusion due to alpha
        phi += alpha * gamma_r * ne * tau_mu_array[mu_0]

        # update population and phi according to event type
        pop_change = update_occupations(mu_0, na, phi)
        na += pop_change[0]
        ne += pop_change[1]
        na_out = pop_change[2]
        phi+= pop_change[3]
        t_tot += tau_mu_array[mu_0]

        # safe na
        na_in_array[ii] = na
        na_out_array[ii] = na_out
        ne_array[ii] = ne
        phi_array[ii] = phi
        t_array[ii] = t_tot

    return t_array, na_out_array, na_in_array, ne_array, phi_array, dtau_list

def bin_laser_timeseries(t_array: np.ndarray, n_array: np.ndarray, bin_size: float) -> (np.ndarray, np.ndarray):
    r"""
    Bin events :math:`n(t_i)` happening on an non-equidistant time grid :math:`t_i` into
    equidistant time bins:

    .. math::
        n_\mathrm{bin}(t) = \sum_{t_i \in [t, t+\Detla t]} n(t_i)

    Args:
        t_array: Array containing non-equidistant times :math:`t_i`.
        n_array: Array containing values at time points :math:`n(t_i)`.
        bin_size: Length of the time bins :math:`\Delta t`.

    Returns:
        Array containing binned times.
        Array containing binned events :math:`n_\mathrm{bin}`.
    """
    t_binned = np.arange(min(t_array), max(t_array), bin_size)
    n_binned = np.zeros_like(t_binned)

    for ii, tt in enumerate(t_binned):
        indices = np.where((t_array > tt) & (t_array <= tt + bin_size))[0]
        n_binned[ii] += np.sum(n_array[indices])
        # print(tt, max(t_array))

    return t_binned, n_binned


def calc_g2_0(n_array: np.ndarray, t_array: np.ndarray, start_ind:int = 0) -> float:
    r"""
    Calculate :math:`g^{(2)}(0) = \frac{\langle n^2 \rangle - \langle n \rangle}{\langle n \rangle}^2` for a timeseries :math:`n(t)`.

    Args:
        n_array: Array containing values at time points :math:`n(t_i)`.
        t_array: Array containing (non-equidistant) times :math:`t_i`.
        start_ind: First index :math:`j` to start at, throws away all values for :math:`i \ll j`.

    Returns:
        :math:`g^{(2)}(0)`.
    """

    dt = np.diff(t_array[start_ind:len(t_array)])

    # Compute the weighted average
    n_avg = (n_array[start_ind:-1] @ dt) / (t_array[-1] - t_array[start_ind])
    n2_avg = (n_array[start_ind:-1]**2 @ dt) / (t_array[-1] - t_array[start_ind])

    return (n2_avg - n_avg) / n_avg**2