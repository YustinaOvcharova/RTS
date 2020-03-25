import numpy as np
import matplotlib.pyplot as plt
import random as rnd
from math import pi


def generate_signal(n: int, Wmax: float, N: int, step: float = 1.0):
    A, fi, w = 0.0, 0.0, 0.0
    maxArgument = step * N
    signal = np.zeros(N, dtype=np.float32)
    t = np.arange(0, maxArgument, step, dtype=np.float32)
    for i in range(n):
        A = rnd.random()
        fi = rnd.uniform(0, 2 * pi)
        w = Wmax / n * (i + 1)
        signal += A * np.sin(w * t + fi)
    return t, signal


def math_expectation(arr):
    sum = 0.0
    for elem in arr:
        sum += elem
    return sum / len(arr)


def dispersion(arr):
    m = math_expectation(arr)
    sum = 0.0
    for elem in arr:
        sum += (elem - m) ** 2
    return sum / len(arr)


def correlation_function(arr1, arr2, M1=None, M2=None) -> np.ndarray:
    if (M1 == None):
        M1 = sum(arr1) / len(arr1)
    if (M2 == None):
        M2 = sum(arr2) / len(arr2)
    N = len(arr1)
    correlation = np.empty(N)
    sum = 0.0
    for tau in range(N - 1):
        sum = 0.0
        for i in range(N - tau):
            sum += (arr1[i] - M1) * (arr2[i + tau] - M2)
        correlation[tau] = sum / (N - tau - 1)
    correlation[N - 1] = correlation[N - 2]
    return correlation

if_name == "_main_":
    n = 10  # n-число гармонік
    Wmax = 1500  # Wmax-гранична частота
    N = 256  # N-кількість дискретних відліків
    t, signal1 = generate_signal(n, Wmax, N, 0.0001)
    t, signal2 = generate_signal(n, Wmax, N, 0.0001)
    # M-мат. очікування, D - дисперсія
    Mx1, Dx1 = signal1.mean(), signal1.var()
    Mx2, Dx2 = signal2.mean(), signal2.var()
    Rxx = correlation_function(signal1, signal1, Mx1, Mx1)
    Rxy = correlation_function(signal1, signal2, Mx1, Mx2)

    print("Signal 1:")
    print("Math expectation Mx1 = ", Mx1)
    print("Dispersion Dx1 = ", Dx1)
    print("Signal 2:")
    print("Math expectatio Mx2 = ", Mx2)
    print("Dispersion Dx2 = ", Dx2)

    fig1, (ax1, ax2) = plt.subplots(2)
    fig1.suptitle('Random signals')
    ax1.set_ylabel('x(t) - signal 1')
    ax1.plot(t, signal1, color='blue', linewidth=0.5)
    ax1.grid(True)
    ax2.set_xlabel('t')
    ax2.set_ylabel('y(t) - signal 2')
    ax2.plot(t, signal2, color='blue', linewidth=0.5)
    ax2.grid(True)

    fig2, (ax3, ax4) = plt.subplots(2)
    fig2.suptitle('Correlation functions')

    ax3.set_ylabel('Rxx')
    ax3.plot(t, Rxx, color='red', linewidth=0.5)
    ax3.grid(True)

    ax4.set_xlabel('tau')
    ax4.set_ylabel('Rxy')
    ax4.plot(t, Rxy, color='red', linewidth=0.5)
    ax4.grid(True)

plt.show()
