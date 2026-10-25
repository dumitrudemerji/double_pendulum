"""
Gymnasium environment for the double pendulum inverted with three integration modes for MPC:

  - 'vanilla':       only RL (replica Lee et al. 2025)
  - 'safety_filter': MPC replaces dangerous RL actions at runtime (Mode A)
  - 'reward_shaper': MPC provides reward shaping during training (Mode B)
  - 'hierarchical':  RL apprende correzioni delta sul setpoint MPC (Mode C)

Observation (Lee et al. Sec. 4.2.2):
  s = [y, sin(th1), cos(th1), sin(th2), cos(th2), ydot, th1d, th2d, tau]
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from models.dynamics import rk4_step
from configs.config import (
    SIM_PARAMS, CONSTRAINTS, EQUILIBRIUM_POINTS, RL_PARAMS
)


class DoublePendulumEnv(gym.Env):
    metadata = {'render_modes': []}

    VALID_MODES = ('vanilla', 'reward_shaper', 'safety_filter', 'hierarchical')

    def __init__(self, mode='vanilla', mpc_controller=None,
                 randomize_tau=True, fixed_tau=None,
                 mpc_shaper_freq=10):
        """
        Args:
            mode: one of the four modes
            mpc_controller: NMPCController, required if mode != 'vanilla'
            randomize_tau: True for training, False for testing
            fixed_tau: fixed tau when randomize_tau=False
            mpc_shaper_freq: in 'reward_shaper' call MPC every N steps
                             (to reduce computational cost)
        """
        super().__init__()
        if mode not in self.VALID_MODES:
            raise ValueError(f"invalid mode: {mode}. Valid: {self.VALID_MODES}")
        if mode != 'vanilla' and mpc_controller is None:
            raise ValueError(f"mode='{mode}' requires mpc_controller")

        self.mode = mode
        self.mpc = mpc_controller
        self.randomize_tau = randomize_tau
        self.fixed_tau = fixed_tau if fixed_tau is not None else 3
        self.mpc_shaper_freq = mpc_shaper_freq

        self.dt_ode = SIM_PARAMS['dt']
        self.dt_ctrl = SIM_PARAMS['control_dt']
        self.n_substeps = int(self.dt_ctrl / self.dt_ode)
        self.max_steps = SIM_PARAMS['max_steps']

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        obs_low = np.array([
            -CONSTRAINTS['y_max'], -1, -1, -1, -1,
            -10.0, -50.0, -50.0, 0.0
        ], dtype=np.float32)
        obs_high = np.array([
            +CONSTRAINTS['y_max'], 1, 1, 1, 1,
            +10.0, +50.0, +50.0, 3.0
        ], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high,
                                             dtype=np.float32)

        self.state = None
        self.tau = None
        self.step_count = 0

    # ---- helpers ----
    def _get_obs(self):
        y, th1, th2, ydot, th1d, th2d = self.state
        return np.array([
            y, np.sin(th1), np.cos(th1),
            np.sin(th2), np.cos(th2),
            ydot, th1d, th2d, float(self.tau)
        ], dtype=np.float32)

    def _denormalize_action(self, a):
        u_max = CONSTRAINTS['u_max']
        a = np.asarray(a, dtype=np.float32).item()
        return float(np.clip(a, -1.0, 1.0) * u_max)

    def _lee_reward(self, u):
        """Reward function of Lee et al. (Eq. 8-9)."""
        y, th1, th2, _, th1d, th2d = self.state
        ep = EQUILIBRIUM_POINTS[self.tau]
        Ru   = np.exp(-0.015 * abs(u))
        Ry   = np.exp(-0.5 * abs(y))
        Rth1 = 0.5 + 0.5 * np.cos(th1 - ep['theta1'])
        Rth2 = 0.5 + 0.5 * np.cos(th2 - ep['theta2'])
        Rd1  = np.exp(-0.02 * abs(th1d))
        Rd2  = np.exp(-0.02 * abs(th2d))
        return Ru * Ry * Rth1 * Rth2 * Rd1 * Rd2

    def _mpc_reward_shaping(self):
        """Mode B: reward shaping based on the NMPC cost."""
        try:
            _, _, _, cost = self.mpc.solve(self.state, self.tau)
            return -float(np.tanh(cost / 100.0))
        except Exception:
            return 0.0

    def _safety_filter(self, a_rl):
        """Mode A: NMPC override when the RL action is dangerous."""
        u_rl = self._denormalize_action(a_rl)
        y, _, _, ydot, _, _ = self.state
        y_warn = 0.85 * CONSTRAINTS['y_max']

        approaching_limit = (
            (y > y_warn and ydot > 0 and u_rl > 0) or
            (y < -y_warn and ydot < 0 and u_rl < 0)
        )
        if approaching_limit:
            try:
                u_mpc, _, _, _ = self.mpc.solve(self.state, self.tau)
                return u_mpc, True
            except Exception:
                return u_rl, False
        return u_rl, False

    def _hierarchical_action(self, a_rl):
        """Mode C: u = u_mpc + delta_rl (delta in [-u_max/3, u_max/3])."""
        try:
            u_mpc, _, _, _ = self.mpc.solve(self.state, self.tau)
        except Exception:
            u_mpc = 0.0
        a_rl = np.asarray(a_rl).item()
        delta = a_rl * (CONSTRAINTS['u_max'] / 3.0)
        return float(np.clip(
            u_mpc + delta, CONSTRAINTS['u_min'], CONSTRAINTS['u_max']
        ))

    # ---- gym API ----
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.tau = (int(self.np_random.integers(0, 4))
                    if self.randomize_tau else self.fixed_tau)

        self.state = np.array([
            self.np_random.uniform(-0.3, 0.3),
            self.np_random.uniform(-np.pi, np.pi),
            self.np_random.uniform(-np.pi, np.pi),
            self.np_random.uniform(-1.0, 1.0),
            self.np_random.uniform(-10.0, 10.0),
            self.np_random.uniform(-20.0, 20.0),
        ])
        self.step_count = 0
        if self.mpc is not None:
            self.mpc.reset()
        return self._get_obs(), {}

    def step(self, action):
        filtered = False
        if self.mode == 'vanilla' or self.mode == 'reward_shaper':
            u = self._denormalize_action(action)
        elif self.mode == 'safety_filter':
            u, filtered = self._safety_filter(action)
        elif self.mode == 'hierarchical':
            u = self._hierarchical_action(action)

        # Dynamics integration (substep RK4)
        for _ in range(self.n_substeps):
            self.state = rk4_step(self.state, u, self.dt_ode)

        # Base reward
        reward = self._lee_reward(u)

        # Reward shaping MPC (Mode B) - called at reduced frequency
        if (self.mode == 'reward_shaper' and
                self.step_count % self.mpc_shaper_freq == 0):
            r_mpc = self._mpc_reward_shaping()
            reward = reward + RL_PARAMS['lambda_mpc'] * r_mpc

        self.step_count += 1
        terminated = False
        truncated = False

        if abs(self.state[0]) > CONSTRAINTS['y_max']:
            terminated = True
            reward -= 10.0

        if self.step_count >= self.max_steps:
            truncated = True

        info = {
            'tau': self.tau,
            'u_applied': u,
            'mpc_filtered': filtered,
        }
        return self._get_obs(), float(reward), terminated, truncated, info
