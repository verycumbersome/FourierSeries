import math
import pyaudio

import numpy as np
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import seaborn as sns

from math import e as e
from math import pi as pi

p = pyaudio.PyAudio()

plt.ion()
fig, (ax1, ax2) = plt.subplots(2)
fig.suptitle('Vertically stacked subplots')

# freqs = [523, 440, 349]
freqs = [1, 4, 15]

def func_aperiodic(x):
    falloff = e ** (-2 * pi * (x ** 2))
    out = sum([math.cos(2 * pi * f * x) for f in freqs])

    return out


def func_periodic(x):
    out = sum([math.cos(f * x) for f in freqs])

    return out


def fourier_transform(freq):
    """ integral(x(t) * e^(-i2(pi)ft)) dx/dt """

    fourier = lambda t: func_aperiodic(t) * (e ** (-2j * pi * freq * t))
    fourier_full = lambda t: fourier(t).real + fourier(t).imag

    out = integrate.quad(fourier_full, -np.inf, np.inf)[0]

    return out


def fourier_series(x, num_iter=1):
    a_0 = (1 / pi) * integrate.quad(func_periodic, -pi, pi)[0]
    coef_a = lambda x: func_periodic(x) * math.cos(n * x)
    coef_b = lambda x: func_periodic(x) * math.sin(n * x)
    a_n = lambda n: (1 / pi) * integrate.quad(coef_a, -pi, pi)[0]
    b_n = lambda n: (1 / pi) * integrate.quad(coef_b, -pi, pi)[0]

    out = a_0 / 2
    for n in range(num_iter):
        out += (a_n(n) * math.cos(n * x)) + (b_n(n) * math.sin(n * x))

    return out


def plot_func(function, span, ax, n=None):
    volume = 0.5     # range [0.0, 1.0]
    fs = 44100       # sampling rate, Hz, must be integer
    duration = 1.0   # in seconds, may be float
    f = 440.0        # sine frequency, Hz, may be float

    x = np.linspace(-(duration / 2), -(duration / 2), fs)
    y = np.vectorize(function)(x, n) if n else np.vectorize(function)(x)

    samples = (y).astype(np.float32)

    print(samples)
    print(len(samples))

    # for paFloat32 sample values must be in range [-1.0, 1.0]
    stream = p.open(format=pyaudio.paFloat32,
                                    channels=1,
                                    rate=fs,
                                    output=True)

    # play. May repeat with different volume values (if done interactively) 
    stream.write(volume*samples)

    stream.stop_stream()
    stream.close()

    return x, y


def play_sound():
    volume = 0.5     # range [0.0, 1.0]
    fs = 44100       # sampling rate, Hz, must be integer
    duration = 1.0   # in seconds, may be float
    f = 440.0        # sine frequency, Hz, may be float

    # generate samples, note conversion to float32 array
    samples = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)

    print(samples)
    print(len(samples))

    # for paFloat32 sample values must be in range [-1.0, 1.0]
    stream = p.open(format=pyaudio.paFloat32,
                                    channels=1,
                                    rate=fs,
                                    output=True)

    # play. May repeat with different volume values (if done interactively) 
    stream.write(volume*samples)

    stream.stop_stream()
    stream.close()


if __name__=="__main__":
    play_sound()

    x, y = plot_func(func_periodic, 50, ax1)
    sns.lineplot(x=x, y=y, ax=ax1)

    print("asdf")
    plt.show()
    plt.pause(10)

    x, y = plot_func(fourier_series, 50, ax2, 20)
    sns.lineplot(x=x, y=y, ax=ax2)

    # num_iter = 0
    # while num_iter < 20:
        # x, y = plot_func(fourier_series, 50, ax2, num_iter)
        # print(y)

        # for line in ax2.lines:
            # line.set_ydata(y)
            # fig.canvas.draw()
            # fig.canvas.flush_events()

        # plt.pause(1)


