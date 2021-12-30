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


def display_func(function):
    x = np.linspace(0, 20, 100)
    y = np.linspace(0, 20, 100)
    grid = np.meshgrid(x, y)

    # Map function across numpy array 
    out = np.vectorize(function)(x)

    plt.plot(out)

    return out

if __name__=="__main__":
    display_func(func_orig)
    # display_func(func_prime)

    plt.show(block=False)
    plt.pause(0.001)
    input("hit[enter] to end.")
    plt.close('all')

