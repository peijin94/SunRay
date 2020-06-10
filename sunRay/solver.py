#import numpy as np 

def rk4_step(f, t0, y0, h):
    k1 = h * f(t0, y0)
    k2 = h * f(t0 + 0.5 * h, y0 + 0.5 * k1)
    k3 = h * f(t0 + 0.5 * h, y0 + 0.5 * k2)
    k4 = h * f(t0 + h, y0 + k3)
    return y0 + (k1 + k2 + k2 + k3 + k3 + k4) / 6

def rk4(f, t0, y0, t1, n):
    vt = [0] * (n + 1)
    vy = [0] * (n + 1)
    h = (t1 - t0) / (n*1.0)
    vt[0] = t = t0
    vy[0] = y = y0
    for i in range(1, n + 1):
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(t + h, y + k3)
        vt[i] = t = t0 + i * h
        vy[i] = y = y + (k1 + k2 + k2 + k3 + k3 + k4) / 6
    return vt, vy