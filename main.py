import math

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import seaborn as sns

fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')
# ax = plt.axes(projection='3d')

# freqs = [523, 440, 349]
freqs = [1, 4, 15]


def func(x):
    out = sum([math.sin(f * x) for f in freqs])

    return out


def fourier_transform(w):
    # x(t) * e^-i2(pi)ft
    fourier = lambda t: (func(t) * (math.e ** (-1j * w * t))).imag
    out = integrate.quad(fourier, -np.inf, np.inf)[0]

    return out


def display_func(func):
    x = np.linspace(0, 20, 100)
    f = np.linspace(0, 30, 100)

    # Map function across numpy array 
    out = np.vectorize(func)(x)
    fourier = np.vectorize(fourier_transform)(f)

    sns.lineplot(x=x, y=out, ax=ax1)
    sns.lineplot(x=f, y=fourier, ax=ax2)


if __name__=="__main__":
    display_func(func)
    # display_func(func_prime)

    plt.autoscale()
    plt.show(block=True)
    plt.pause(0.001)
    input("hit[enter] to end.")
    plt.close('all')

