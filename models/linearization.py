"""
Linearization of the system around the equilibrium points
and computation of the LQR gain for the terminal cost of the MPC.
"""
import numpy as np
import casadi as ca
from scipy.linalg import solve_continuous_are

from models.dynamics import dynamics_casadi
from configs.config import EQUILIBRIUM_POINTS, MPC_PARAMS


def linearize_at_equilibrium(tau):
    """Linearize around the equilibrium point. Returns A, B, x_eq."""
    ep = EQUILIBRIUM_POINTS[tau]
    x_eq = np.array([0.0, ep['theta1'], ep['theta2'], 0.0, 0.0, 0.0])
    u_eq = 0.0

    f = dynamics_casadi()
    x_sym = ca.SX.sym('x', 6)
    u_sym = ca.SX.sym('u', 1)
    f_expr = f(x_sym, u_sym)

    A_func = ca.Function('A', [x_sym, u_sym], [ca.jacobian(f_expr, x_sym)])
    B_func = ca.Function('B', [x_sym, u_sym], [ca.jacobian(f_expr, u_sym)])

    A = np.array(A_func(x_eq, u_eq))
    B = np.array(B_func(x_eq, u_eq))
    return A, B, x_eq


def compute_lqr_terminal_cost(tau):
    """Solve the continuous Riccati equation. Returns P, K, x_eq."""
    A, B, x_eq = linearize_at_equilibrium(tau)
    Q = np.diag(MPC_PARAMS['Q_diag'])
    R = np.array([[MPC_PARAMS['R_weight']]])

    try:
        P = solve_continuous_are(A, B, Q, R)
        K = np.linalg.solve(R, B.T @ P)
    except Exception as e:
        print(f"[WARN] Riccati not solvable for tau={tau}: {e}")
        P = Q.copy()
        K = np.zeros((1, 6))
    return P, K, x_eq


def precompute_all_terminal_costs():
    """Pre-compute P_LQR for all 4 EP."""
    terminal_costs = {}
    for tau in EQUILIBRIUM_POINTS.keys():
        P, K, x_eq = compute_lqr_terminal_cost(tau)
        terminal_costs[tau] = {'P': P, 'K': K, 'x_eq': x_eq}
        print(f"[INFO] EP{tau} ({EQUILIBRIUM_POINTS[tau]['name']}): "
              f"trace(P)={np.trace(P):.2f}")
    return terminal_costs
