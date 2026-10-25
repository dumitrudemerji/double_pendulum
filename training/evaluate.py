"""
Evaluation script: test the RL controller on the 12 transition scenarios
from the paper Lee et al. 2025 and test the recovery features with
external disturbances.

Esempi:
  python3 -m training.evaluate --model models_saved/tqc_safety_filter/final.zip --mode safety_filter
  python3 -m training.evaluate --model models_saved/tqc_safety_filter/final.zip --mode safety_filter --disturbance
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sb3_contrib import TQC
from controllers.environment import DoublePendulumEnv
from controllers.nmpc import NMPCController
from configs.config import EQUILIBRIUM_POINTS, SIM_PARAMS, CONSTRAINTS

# Sequence of 12 transition scenarios (Fig. 11 of the Lee paper)
TRANSITION_SEQUENCE = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (0, 2), (2, 0), (0, 3), (3, 2),
    (2, 1), (1, 3), (3, 1), (1, 0),
]

def wrap_angle(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def evaluate_transitions(model, mode, mpc=None, episode_duration=6.0,
                          render=False):
    """Execute the sequence of 12 transitions and collect the data."""
    env = DoublePendulumEnv(
        mode=mode, mpc_controller=mpc,
        randomize_tau=False, fixed_tau=0
    )
    obs, _ = env.reset(seed=0)

    # Initialize in EP0 (down-down) with exact state
    env.state = np.array([0.0, -np.pi, 0.0, 0.0, 0.0, 0.0])

    log = {'t': [], 'y': [], 'th1': [], 'th2': [],
           'ydot': [], 'th1d': [], 'th2d': [],
           'u': [], 'tau': [], 'reward': []}

    steps_per_transition = int(episode_duration / SIM_PARAMS['control_dt'])
    t = 0.0
    dt = SIM_PARAMS['control_dt']

    for (tau_from, tau_to) in TRANSITION_SEQUENCE:
        env.tau = tau_to  # imposta nuovo target
        if mpc is not None:
            mpc.reset()

        for _ in range(steps_per_transition):
            obs = env._get_obs()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            log['t'].append(t)
            log['y'].append(env.state[0])
            log['th1'].append(wrap_angle(env.state[1]))
            log['th2'].append(wrap_angle(env.state[2]))
            log['ydot'].append(env.state[3])
            log['th1d'].append(env.state[4])
            log['th2d'].append(env.state[5])
            log['u'].append(info['u_applied'])
            log['tau'].append(env.tau)
            log['reward'].append(reward)
            t += dt
            if terminated:
                print(f"[WARN] Early termination at t={t:.2f}s "
                      f"(tau {tau_from}->{tau_to})")
                # Reset state near the target to continue the test
                ep = EQUILIBRIUM_POINTS[tau_to]
                env.state = np.array([0.0, ep['theta1'], ep['theta2'],
                                       0.0, 0.0, 0.0])

    return {k: np.array(v) for k, v in log.items()}


def evaluate_recovery(model, mode, mpc=None, target_tau=3,
                       disturbance_time=6.0, disturbance_magnitude=15.0,
                       total_time=15.0):
    """Evaluate the recovery capability by applying a strong disturbance."""
    env = DoublePendulumEnv(
        mode=mode, mpc_controller=mpc,
        randomize_tau=False, fixed_tau=target_tau
    )
    obs, _ = env.reset(seed=42)

    # Start near EP0 and perform swing-up towards target_tau
    env.state = np.array([0.0, -np.pi, 0.0, 0.0, 0.0, 0.0])

    log = {'t': [], 'y': [], 'th1': [], 'th2': [],
           'u': [], 'disturbed': []}

    n_steps = int(total_time / SIM_PARAMS['control_dt'])
    t = 0.0
    dt = SIM_PARAMS['control_dt']
    disturbance_step = int(disturbance_time / dt)

    for step in range(n_steps):
        obs = env._get_obs()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        # Apply disturbance: impulse in angular velocity
        disturbed = False
        if step == disturbance_step:
            env.state[4] += disturbance_magnitude  # th1d
            env.state[5] += disturbance_magnitude  # th2d
            disturbed = True
            print(f"[INFO] Disturbance applied at t={t:.2f}s")

        log['t'].append(t)
        log['y'].append(env.state[0])
        log['th1'].append(wrap_angle(env.state[1]))
        log['th2'].append(wrap_angle(env.state[2]))
        log['u'].append(info['u_applied'])
        log['disturbed'].append(disturbed)
        t += dt

        if terminated:
            print(f"[WARN] Recovery failed at t={t:.2f}s")
            break

    return {k: np.array(v) for k, v in log.items()}


def plot_transitions(log, mode_name, save_path=None):
    fig, axes = plt.subplots(5, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(log['t'], log['y'], 'b-')
    axes[0].axhline(+CONSTRAINTS['y_max'], color='k', ls='--', alpha=0.4)
    axes[0].axhline(-CONSTRAINTS['y_max'], color='k', ls='--', alpha=0.4)
    axes[0].set_ylabel('y [m]')
    axes[0].set_title(f'Transition Control - mode: {mode_name}')
    axes[0].grid()

    axes[1].plot(log['t'], log['th1'], 'r-', label=r'$\theta_1$')
    axes[1].set_ylabel('Angle [rad]')
    axes[1].legend()
    axes[1].grid()

    axes[2].plot(log['t'], log['th2'], 'g-', label=r'$\theta_2$')
    axes[2].set_ylabel('Angle [rad]')
    axes[2].legend()
    axes[2].grid()

    axes[3].plot(log['t'], log['u'], 'k-')
    axes[3].set_ylabel(r'u [m/s²]')
    axes[3].set_ylim([CONSTRAINTS['u_min'] * 1.1, CONSTRAINTS['u_max'] * 1.1])
    axes[3].grid()

    axes[4].step(log['t'], log['tau'], 'm-', where='post')
    axes[4].set_ylabel('Target EP')
    axes[4].set_yticks([0, 1, 2, 3])
    axes[4].set_xlabel('Time [s]')
    axes[4].grid()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='svg')
        print(f"[INFO] Plot saved: {save_path}")
    return fig


def plot_recovery(log, mode_name, save_path=None):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    dist_idx = np.argmax(log['disturbed']) if log['disturbed'].any() else None
    dist_t = log['t'][dist_idx] if dist_idx is not None else None

    axes[0].plot(log['t'], log['y'], 'b-')
    if dist_t: axes[0].axvline(dist_t, color='r', ls='--', alpha=0.7)
    axes[0].set_ylabel('y [m]')
    axes[0].set_title(f'Recovery Test - mode: {mode_name}')

    axes[1].plot(log['t'], log['th1'], 'r-', label=r'$\theta_1$')
    axes[1].plot(log['t'], log['th2'], 'g-', label=r'$\theta_2$')
    if dist_t: axes[1].axvline(dist_t, color='r', ls='--', alpha=0.7)
    axes[1].set_ylabel('Angle [rad]')
    axes[1].legend()

    axes[2].plot(log['t'], log['u'], 'k-')
    if dist_t: axes[2].axvline(dist_t, color='r', ls='--', alpha=0.7)
    axes[2].set_ylabel(r'u [m/s²]')
    axes[2].set_xlabel('Time [s]')

    plt.grid()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='svg')
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument(
        '--mode',
        choices=['vanilla', 'reward_shaper', 'safety_filter', 'hierarchical'],
        required=False, default='vanilla',
    )
    parser.add_argument('--disturbance', action='store_true',
                        help='Esegui recovery test invece di transition test')
    parser.add_argument('--out', type=str, default='./eval_output')
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True, parents=True)

    print(f"[INFO] Loading model: {args.model}")
    model = TQC.load(args.model)

    mpc = None
    if args.mode != 'vanilla':
        mpc = NMPCController()

    if args.disturbance:
        log = evaluate_recovery(model, args.mode, mpc)
        plot_recovery(log, args.mode, out / f'recovery_{args.mode}.svg')
    else:
        log = evaluate_transitions(model, args.mode, mpc)
        plot_transitions(log, args.mode, out / f'transitions_{args.mode}.svg')
    
    plt.show()
