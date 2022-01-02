import math

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt

fig = plt.figure()
# ax = plt.axes(projection='3d')


def func_orig(x):
    out = math.cos(523 * x)
    out += math.cos(440 * x)
    out += math.cos(349 * x)

    return out


def func_prime(x):
    out = -math.sin(x)

    return out


def fourier_transform(func, f, t):
    # x(t) * e^-i2(pi)ft
    out = func(t) * math.e ** (2 * math.pi * f * t)

    integrate.quad(f1, np.inf, -np.inf)

    return out


def display_func(func):
    x = np.linspace(0, 20, 100)
    t = np.linspace(-1, 1, 100)

    # Map function across numpy array 
    out = np.vectorize(func)(x)
    fourier = np.vectorize(fourier_transform)(func, 2, t)

    # plt.plot(out)
    plt.plot(fourier)


if __name__=="__main__":
    display_func(func_orig)
    # display_func(func_prime)

    plt.show(block=False)
    plt.pause(0.001)
    input("hit[enter] to end.")
    plt.close('all')

