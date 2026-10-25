"""
Global configuration for the double pendulum inverted system.
Parameters taken from Lee et al. 2025 (Table 1).
"""
import numpy as np

PHYSICAL_PARAMS = {
    'm1': 0.2351, 'I1': 0.0012, 'l1': 0.0667, 'L1': 0.1645, 'c1': 4.5116e-4,
    'm2': 0.1452, 'I2': 0.0010, 'l2': 0.1288, 'c2': 2.9198e-4,
    'g': 9.81,
}

CONSTRAINTS = {
    'y_max': 0.4,
    'u_max': 15.0,
    'u_min': -15.0,
}

EQUILIBRIUM_POINTS = {
    0: {'theta1': -np.pi, 'theta2': 0.0,     'name': 'EP0_DownDown'},
    1: {'theta1': -np.pi, 'theta2': -np.pi,  'name': 'EP1_DownUp'},
    2: {'theta1': 0.0,    'theta2': -np.pi,  'name': 'EP2_UpDown'},
    3: {'theta1': 0.0,    'theta2': 0.0,     'name': 'EP3_UpUp'},
}

SIM_PARAMS = {
    'dt': 0.001,
    'control_dt': 0.01,
    'episode_length': 10.0,
    'max_steps': 800,
}

MPC_PARAMS = {
    'N_horizon': 50,
    'dt_mpc': 0.01,
    'Q_diag': [1200.0, 500.0, 500.0, 100.0, 10.0, 10.0],
    # 'Q_diag': [10.0, 50.0, 50.0, 1.0, 1.0, 1.0],
    'R_weight': 0.006, # 0.001
    'use_lqr_terminal': True,
}

RL_PARAMS = {
    'algorithm': 'TQC',
    'learning_rate': 3e-4,
    'gamma': 0.99,
    'buffer_size': int(1.5e6),
    'batch_size': 256,
    'tau': 0.005,
    'policy_layers': [400, 300],
    'critic_layers': [512, 512, 512],
    'n_critics': 3,
    'n_quantiles': 25,
    'total_timesteps': int(2.5e6),
    'lambda_mpc': 0.3,
}
