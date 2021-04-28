# Adapted from "Neural ODE Processes" by Alexander Norcliffe, Cristian Bodnar,
# Ben Day, Jacob Moss, Pietro Liò

import numpy as np
import torch
from scipy.integrate import odeint
from torch.utils.data import Dataset
from tqdm import tqdm


class SimTooLongException(Exception):
    pass


class MarkovJumpProcess:
    """
    Implements a generic markov jump process and algorithms for simulating it.
    It is an abstract class, it needs to be inherited by a concrete implementation.
    """

    def __init__(self, init, params):

        self.state = np.asarray(init)
        self.params = np.asarray(params)
        self.time = 0.0

    def _calc_propensities(self):
        raise NotImplementedError(
            'This is an abstract method and should be implemented in a subclass.')

    def _do_reaction(self, reaction):
        raise NotImplementedError(
            'This is an abstract method and should be implemented in a subclass.')

    def sim_steps(self, num_steps):
        """Simulates the process with the gillespie algorithm for a specified number of steps."""

        times = [self.time]
        states = [self.state.copy()]

        for _ in range(num_steps):

            rates = self.params * self._calc_propensities()
            total_rate = rates.sum()

            if total_rate == 0:
                self.time = float('inf')
                break

            self.time += np.random.exponential(scale=1/total_rate)

            reaction = self.discrete_sample(rates / total_rate)[0]
            self._do_reaction(reaction)

            times.append(self.time)
            states.append(self.state.copy())

        return times, np.array(states)

    def sim_time(self, dt, duration, max_n_steps=float('inf')):
        """Simulates the process with the gillespie algorithm for a specified time duration."""

        num_rec = int(duration / dt) + 1
        states = np.zeros([num_rec, self.state.size])
        cur_time = self.time
        n_steps = 0

        for i in range(num_rec):

            while cur_time > self.time:

                rates = self.params * self._calc_propensities()
                total_rate = rates.sum()

                if total_rate == 0:
                    self.time = float('inf')
                    break

                exp_scale = max(1/total_rate, 1e-3)
                self.time += np.random.exponential(scale=exp_scale)

                reaction = np.random.multinomial(1, rates / total_rate)
                reaction = np.argmax(reaction)
                self._do_reaction(reaction)

                n_steps += 1
                if n_steps > max_n_steps:
                    raise SimTooLongException(max_n_steps)

            states[i] = self.state.copy()
            cur_time += dt

        return np.array(states)


class LotkaVolterra(MarkovJumpProcess):
    """Implements the lotka-volterra population model."""

    def _calc_propensities(self):

        x, y = self.state
        xy = x * y
        return np.array([xy, x, y, xy])

    def _do_reaction(self, reaction):

        if reaction == 0:
            self.state[0] += 1
        elif reaction == 1:
            self.state[0] -= 1
        elif reaction == 2:
            self.state[1] += 1
        elif reaction == 3:
            self.state[1] -= 1
        else:
            raise ValueError('Unknown reaction.')


class StochasticLotkaVolteraData(Dataset):
    """
    Dataset of time-seires sampled from a Lotka-Voltera model
    ----------
    amplitude_range : tuple of float
            Defines the range from which the amplitude (i.e. a) of the sine function
            is sampled.
    shift_range : tuple of float
            Defines the range from which the shift (i.e. b) of the sine function is
            sampled.
    num_samples : int
            Number of samples of the function contained in dataset.
    num_points : int
            Number of points at which to evaluate f(x) for x in [-pi, pi].
    """

    def __init__(self, initial_X=50, initial_Y=100,
                 num_samples=1000, dt=0.2):
        self.initial_X = initial_X
        self.initial_Y = initial_Y
        self.num_samples = num_samples
        self.x_dim = 1
        self.y_dim = 2
        self.dt = dt

        self.init = [self.initial_X, self.initial_Y]
        self.params = [0.01, 0.5, 1.0, 0.01]
        self.duration = 30

        # Generate data
        self.data = []
        print("Creating dataset...", flush=True)

        removed = 0
        for samples in range(num_samples):
            lv = LotkaVolterra(self.init, self.params)
            states = lv.sim_time(dt, self.duration)
            times = torch.linspace(0.0, self.duration,
                                   int(self.duration / dt) + 1)
            times = times.unsqueeze(1)

            # Ignore outlier populations
            if np.max(states) > 600:
                removed += 1
                continue

            # Scale the population ranges to be closer to the real model
            states = torch.FloatTensor(states) * 1/100
            times = times * 1/20
            self.data.append((times, states))

        self.num_samples -= removed

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples


class DeterministicLotkaVolteraData(Dataset):
    """
    Dataset of Lotka-Voltera time series.
      Populations (u,v) evolve according to
            u' = \alpha u - \beta u v
            v' = \delta uv - \gamma v
      with the dataset sampled either with (u_0, v_0) fixed and (\alpha, \beta,
      \gamma, \delta) varied, or varying the initial populations for a fixed set
      of greeks.
    If initial values for (u,v) are provided then the greeks are sampled from
            (0.9,0.05,1.25,0.5) to (1.1,0.15,1.75,1.0)
    if values are provided for the greeks then (u_0 = v_0) is sampled from
            (0.5) to (2.0)
    if both are provided, defaults to initial population mode (greeks vary)
    ----------
    initial_u	: int
            fixed initial value for u
    initial_v	: int
            fixed initial value for v
    fixed_alpha : int
            fixed initial value for \alpha
    fixed_beta	: int
            fixed initial value for \beta
    fixed_gamma : int
            fixed initial value for \gamme
    fixed_delta : int
            fixed initial value for \delta
    end_time : float
            the final time (simulation runs from 0 to end_time)
    steps : int
            how many time steps to take from 0 to end_time
    num_samples : int
            Number of samples of the function contained in dataset.
    """

    def __init__(self, initial_u=None, initial_v=None,
                 alpha=None, beta=None, gamma=None, delta=None,
                 num_samples=1000, steps=150, end_time=15):

        if initial_u is None:
            self.mode = 'greek'
            self.alpha = alpha
            self.beta = beta
            self.gamma = gamma
            self.delta = delta
        else:
            self.mode = 'population'
            self.initial_u = initial_u
            self.initial_v = initial_v

        print('Lotka-Voltera is in {self.mode} mode.')

        self.num_samples = num_samples
        self.steps = steps
        self.end_time = end_time

        # Generate data
        self.data = []
        print("Creating dataset...", flush=True)

        removed = 0
        for samples in tqdm(range(num_samples)):
            times, states = self.generate_ts()
            # normalise times
            times = torch.FloatTensor(times) / 10
            times = times.unsqueeze(1)

            states = torch.FloatTensor(states)
            if self.mode == 'population':
                states = states / 100
            #states = torch.cat((states, times), dim=-1)

            self.data.append((times, states))

        self.num_samples -= removed

    def generate_ts(self):
        if self.mode == 'population':
            X_0 = np.array([self.initial_u, self.initial_v])
            a = np.random.uniform(0.9, 1.1)
            b = np.random.uniform(0.05, 0.15)
            c = np.random.uniform(1.25, 1.75)
            d = np.random.uniform(0.5, 1.0)
        else:
            equal_pop = np.random.uniform(0.25, 1.)
            X_0 = np.array([2*equal_pop, equal_pop])
            a, b, c, d = self.alpha, self.beta, self.gamma, self.delta

        def dX_dt(X, t=0):
            """ Return the growth rate of fox and rabbit populations. """
            return np.array([a*X[0] - b*X[0]*X[1],
                             -c*X[1] + d*X[0]*X[1]])

        t = np.linspace(0, self.end_time, self.steps)
        X = odeint(dX_dt, X_0, t)

        return t, X

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples
