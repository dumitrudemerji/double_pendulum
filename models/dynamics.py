"""
Dynamic model of the inverted double pendulum.
Implements the equations of Lee et al. 2025 (Eq. 1-6) using Euler-Lagrange.

NumPy version (for simulation) and CasADi version (for MPC).
"""
import numpy as np
import casadi as ca
from configs.config import PHYSICAL_PARAMS as P


def _h_constants():
    g = P['g']
    h1 = P['m1'] * P['l1'] + P['m2'] * P['L1']
    h2 = P['m2'] * P['l2']
    h3 = P['I1'] + P['m1'] * P['l1']**2 + P['m2'] * P['L1']**2
    h4 = P['m2'] * P['L1'] * P['l2']
    h5 = g * (P['m1'] * P['l1'] + P['m2'] * P['L1'])
    h6 = P['I2'] + P['m2'] * P['l2']**2
    h7 = g * P['m2'] * P['l2']
    return h1, h2, h3, h4, h5, h6, h7


H1, H2, H3, H4, H5, H6, H7 = _h_constants()


def dynamics_numpy(x, u):
    """xdot = f(x, u). State: [y, th1, th2, ydot, th1d, th2d]. Input: u (acc)."""
    y, th1, th2, ydot, th1d, th2d = x

    n1 = H1 * np.cos(th1) + H2 * np.cos(th1 + th2)
    n2 = H2 * np.cos(th1 + th2)
    m11 = H3 + H6 + 2 * H4 * np.cos(th2)
    m12 = H6 + H4 * np.cos(th2)
    m21 = m12
    m22 = H6
    r1 = (-H4 * np.sin(th2) * (2 * th1d * th2d + th2d**2)
          - H5 * np.sin(th1)
          - H7 * np.sin(th1 + th2)
          + P['c1'] * th1d)
    r2 = (H4 * np.sin(th2) * th1d**2
          - H7 * np.sin(th1 + th2)
          + P['c2'] * th2d)
    Phi = m11 * m22 - m12 * m21
    th1dd = ((-m22 * n1 + m12 * n2) * u + (-m22 * r1 + m12 * r2)) / Phi
    th2dd = ((m21 * n1 - m11 * n2) * u + (m21 * r1 - m11 * r2)) / Phi
    return np.array([ydot, th1d, th2d, u, th1dd, th2dd])


def rk4_step(x, u, dt):
    """Runge-Kutta 4th order integrator."""
    k1 = dynamics_numpy(x, u)
    k2 = dynamics_numpy(x + 0.5 * dt * k1, u)
    k3 = dynamics_numpy(x + 0.5 * dt * k2, u)
    k4 = dynamics_numpy(x + dt * k3, u)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def dynamics_casadi():
    """Create the symbolic function f(x, u) using CasADi."""
    x = ca.SX.sym('x', 6)
    u = ca.SX.sym('u', 1)
    y, th1, th2, ydot, th1d, th2d = x[0], x[1], x[2], x[3], x[4], x[5]

    n1 = H1 * ca.cos(th1) + H2 * ca.cos(th1 + th2)
    n2 = H2 * ca.cos(th1 + th2)
    m11 = H3 + H6 + 2 * H4 * ca.cos(th2)
    m12 = H6 + H4 * ca.cos(th2)
    m21 = m12
    m22 = H6
    r1 = (-H4 * ca.sin(th2) * (2 * th1d * th2d + th2d**2)
          - H5 * ca.sin(th1)
          - H7 * ca.sin(th1 + th2)
          + P['c1'] * th1d)
    r2 = (H4 * ca.sin(th2) * th1d**2
          - H7 * ca.sin(th1 + th2)
          + P['c2'] * th2d)
    Phi = m11 * m22 - m12 * m21
    th1dd = ((-m22 * n1 + m12 * n2) * u + (-m22 * r1 + m12 * r2)) / Phi
    th2dd = ((m21 * n1 - m11 * n2) * u + (m21 * r1 - m11 * r2)) / Phi

    xdot = ca.vertcat(ydot, th1d, th2d, u, th1dd, th2dd)
    return ca.Function('f', [x, u], [xdot], ['x', 'u'], ['xdot'])


def rk4_step_casadi(f, dt):
    """Symbolic RK4 integrator for multiple shooting."""
    x = ca.SX.sym('x', 6)
    u = ca.SX.sym('u', 1)
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    x_next = x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return ca.Function('rk4', [x, u], [x_next])
