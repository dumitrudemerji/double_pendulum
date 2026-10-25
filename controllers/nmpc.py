"""
NMPC controller for the inverted double pendulum (multiple shooting + IPOPT).

Supports two variants, activable from MPC_PARAMS['use_integral_state']:

  STANDARD (n=6): physical state
    [y, th1, th2, ydot, th1d, th2d]

  AUGMENTED (n=7): aggiunge integrale di y per offset-free tracking
    [y, th1, th2, ydot, th1d, th2d, e_y]   con e_y_dot = y

L'augmented permette all'MPC di "ricordare" il drift accumulato del carrello
e generare azione correttiva — analogo al termine I di un PI sulla posizione.

Problem:
    min  sum_{k=0..N-1} l(x_k, u_k) + V_N(x_N)
    s.t. x_{k+1} = f_d(x_k, u_k)
         u_min <= u_k <= u_max
         |y_k| <= y_max
         |e_y_k| <= integral_clamp        (anti-windup, solo augmented)
         x_0 = x_init
"""
import numpy as np
import casadi as ca

from models.dynamics import (
    dynamics_casadi, rk4_step_casadi,)
    # dynamics_casadi_aug, rk4_step_casadi_aug,)

from models.linearization import precompute_all_terminal_costs
from configs.config import MPC_PARAMS, CONSTRAINTS, EQUILIBRIUM_POINTS


class NMPCController:
    def __init__(self, verbose=False):
        self.N = MPC_PARAMS['N_horizon']
        self.dt = MPC_PARAMS['dt_mpc']
        self.use_aug = MPC_PARAMS.get('use_integral_state', False)
        self.nx = 7 if self.use_aug else 6   # state dimension

        # Q matrix
        Q_phys = np.diag(MPC_PARAMS['Q_diag'])    # 6x6
        if self.use_aug:
            self.Q = np.zeros((7, 7))
            self.Q[:6, :6] = Q_phys
            self.Q[6, 6] = MPC_PARAMS['Q_integral']
        else:
            self.Q = Q_phys
        self.R = np.array([[MPC_PARAMS['R_weight']]])

        # Terminal cost: precomputed on the physical system (6x6)
        # If augmented, we extend it by adding weight on the integral
        self.terminal_costs = precompute_all_terminal_costs()
        if self.use_aug:
            for tau, ep in self.terminal_costs.items():
                P6 = ep['P']
                P7 = np.zeros((7, 7))
                P7[:6, :6] = P6
                P7[6, 6] = MPC_PARAMS['Q_integral']  # same weight as stage cost
                ep['P'] = P7

        # Disctrete dynamics f_d(x, u) = x_next
        
        
        # if self.use_aug:
        #     f = dynamics_casadi_aug()
        #     self.f_d = rk4_step_casadi_aug(f, self.dt)
        # else:
        f = dynamics_casadi()
        self.f_d = rk4_step_casadi(f, self.dt)

        # Costruisci un solver per ogni EP
        self.solvers = {}
        self.lbx_list = {}
        self.ubx_list = {}
        for tau in EQUILIBRIUM_POINTS.keys():
            self._build_solver(tau, verbose=verbose)

        self.last_solution = None
        # Integral state (solo nella modalità augmented)
        self._e_y = 0.0

    def _build_solver(self, tau, verbose=False):
        N = self.N
        nx = self.nx
        ep = self.terminal_costs[tau]
        x_eq6 = ep['x_eq']               # physical equilibrium (6,)
        if self.use_aug:
            x_eq = np.concatenate([x_eq6, [0.0]])   # e_y* = 0
        else:
            x_eq = x_eq6
        P_term = ep['P']

        X = ca.SX.sym('X', nx, N + 1)
        U = ca.SX.sym('U', 1, N)
        x0_param = ca.SX.sym('x0', nx)

        def angle_err(th, th_ref):
            return ca.atan2(ca.sin(th - th_ref), ca.cos(th - th_ref))

        def stage_err(xk):
            """Wrapped error for the angles, linear for the others."""
            err6 = ca.vertcat(
                xk[0] - x_eq[0],
                angle_err(xk[1], x_eq[1]),
                angle_err(xk[2], x_eq[2]),
                xk[3] - x_eq[3],
                xk[4] - x_eq[4],
                xk[5] - x_eq[5],
            )
            if self.use_aug:
                return ca.vertcat(err6, xk[6] - x_eq[6])
            return err6

        cost = 0
        g = [X[:, 0] - x0_param]

        for k in range(N):
            x_err = stage_err(X[:, k])
            cost += ca.mtimes([x_err.T, self.Q, x_err])
            cost += ca.mtimes([U[:, k].T, self.R, U[:, k]])

            x_next_pred = self.f_d(X[:, k], U[:, k])
            g.append(X[:, k + 1] - x_next_pred)

        # Terminal cost
        x_err_N = stage_err(X[:, N])
        cost += ca.mtimes([x_err_N.T, P_term, x_err_N])

        # Setup NLP
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        nlp = {'x': opt_vars, 'p': x0_param,
               'f': cost, 'g': ca.vertcat(*g)}
        opts = {
            'ipopt.print_level': 0 if not verbose else 3,
            'print_time': verbose,
            'ipopt.max_iter': 100,
            'ipopt.tol': 1e-4,
            'ipopt.acceptable_tol': 1e-3,
        }
        solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

        # Bounds
        n_x_total = nx * (N + 1)
        n_u = N
        lbx = -np.inf * np.ones(n_x_total + n_u)
        ubx = +np.inf * np.ones(n_x_total + n_u)

        # State constraints for each time step k
        clamp = MPC_PARAMS.get('integral_clamp', np.inf)
        for k in range(N + 1):
            base = nx * k
            # y limited
            lbx[base + 0] = -CONSTRAINTS['y_max']
            ubx[base + 0] = +CONSTRAINTS['y_max']
            # e_y limited (anti-windup) only if augmented
            if self.use_aug:
                lbx[base + 6] = -clamp
                ubx[base + 6] = +clamp
        # Input constraints
        lbx[n_x_total:] = CONSTRAINTS['u_min']
        ubx[n_x_total:] = CONSTRAINTS['u_max']

        self.solvers[tau] = solver
        self.lbx_list[tau] = lbx
        self.ubx_list[tau] = ubx

    def solve(self, x_current, tau):
        """
        Solve MPC. Accepts x_current of size 6 (the physical state).
        Internally, in augmented mode, adds the estimated integral.

        Returns: (u, X_pred [nx,N+1], U_pred [1,N], cost)
        """
        N = self.N
        nx = self.nx

        # Increase the state if necessary
        if self.use_aug:
            # Update integral with trapezoidal method
            clamp = MPC_PARAMS.get('integral_clamp', np.inf)
            self._e_y = float(np.clip(
                self._e_y + x_current[0] * self.dt,
                -clamp, clamp
            ))
            x_init_state = np.concatenate([x_current[:6], [self._e_y]])
        else:
            x_init_state = np.asarray(x_current[:6])

        n_x_total = nx * (N + 1)
        if self.last_solution is not None:
            x_init_guess = self.last_solution
        else:
            x_init_guess = np.concatenate([
                np.tile(x_init_state, N + 1),
                np.zeros(N),
            ])

        n_g = nx * (N + 1)
        sol = self.solvers[tau](
            x0=x_init_guess,
            p=x_init_state,
            lbx=self.lbx_list[tau],
            ubx=self.ubx_list[tau],
            lbg=np.zeros(n_g),
            ubg=np.zeros(n_g),
        )

        opt = np.array(sol['x']).flatten()
        X_pred = opt[:n_x_total].reshape((N + 1, nx)).T
        U_pred = opt[n_x_total:].reshape((1, N))
        cost = float(sol['f'])

        # Warm start: shift the solution by one step
        shifted = np.concatenate([
            opt[nx:n_x_total],
            opt[n_x_total - nx:n_x_total],
            opt[n_x_total + 1:],
            opt[-1:],
        ])
        self.last_solution = shifted

        return float(U_pred[0, 0]), X_pred, U_pred, cost

    def reset(self):
        """Reset the internal state (warm start + integral)."""
        self.last_solution = None
        self._e_y = 0.0