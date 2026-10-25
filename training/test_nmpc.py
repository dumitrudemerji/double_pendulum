"""
Test standalone del controllore NMPC: verifica che riesca a fare swing-up
e stabilizzazione senza aiuto dell'RL.

Utile per:
1. Verificare che il modello CasADi sia corretto
2. Testare le 4 task di stabilizzazione (uno per ogni EP)
3. Generare traiettorie di riferimento per behavioral cloning
"""
import numpy as np
import matplotlib.pyplot as plt

from controllers.nmpc import NMPCController
from models.dynamics import rk4_step
from configs.config import (
    EQUILIBRIUM_POINTS, SIM_PARAMS, CONSTRAINTS
)


def run_nmpc_only(target_tau=3, total_time=10.0, x0=None, verbose=False):
    """Esegue solo il controllore NMPC verso il target EP."""
    mpc = NMPCController(verbose=verbose)
    if x0 is None:
        # Parti da EP0 (down-down)
        x0 = np.array([0.0, -np.pi, 0.0, 0.0, 0.0, 0.0])

    state = x0.copy()
    dt_ode = SIM_PARAMS['dt']
    dt_ctrl = SIM_PARAMS['control_dt']
    n_substeps = int(dt_ctrl / dt_ode)
    n_steps = int(total_time / dt_ctrl)

    log = {'t': [], 'y': [], 'th1': [], 'th2': [], 'u': [], 'cost': []}
    t = 0.0

    for _ in range(n_steps):
        try:
            u, _, _, cost = mpc.solve(state, target_tau)
        except Exception as e:
            print(f"[ERR] Solver fallito a t={t:.2f}: {e}")
            break

        log['t'].append(t)
        log['y'].append(state[0])
        log['th1'].append(state[1])
        log['th2'].append(state[2])
        log['u'].append(u)
        log['cost'].append(cost)

        for _ in range(n_substeps):
            state = rk4_step(state, u, dt_ode)
        t += dt_ctrl

        if abs(state[0]) > CONSTRAINTS['y_max']:
            print(f"[WARN] Limite rail superato a t={t:.2f}s")
            break

    return {k: np.array(v) for k, v in log.items()}


def plot_nmpc_run(log, target_tau, save_path=None):
    fig, axes = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    ep_name = EQUILIBRIUM_POINTS[target_tau]['name']

    axes[0].plot(log['t'], log['y'])
    axes[0].axhline(+CONSTRAINTS['y_max'], color='k', ls='--', alpha=0.3)
    axes[0].axhline(-CONSTRAINTS['y_max'], color='k', ls='--', alpha=0.3)
    axes[0].set_ylabel('y [m]')
    axes[0].set_title(f'NMPC Standalone -> {ep_name}')

    th1_ref = EQUILIBRIUM_POINTS[target_tau]['theta1']
    th2_ref = EQUILIBRIUM_POINTS[target_tau]['theta2']
    axes[1].plot(log['t'], log['th1'], label=r'$\theta_1$')
    axes[1].axhline(th1_ref, color='b', ls=':', alpha=0.5)
    axes[1].plot(log['t'], log['th2'], label=r'$\theta_2$')
    axes[1].axhline(th2_ref, color='orange', ls=':', alpha=0.5)
    axes[1].set_ylabel('Angle [rad]')
    axes[1].legend()

    axes[2].plot(log['t'], log['u'])
    axes[2].set_ylabel('u [m/s²]')
    axes[2].set_ylim([CONSTRAINTS['u_min'] * 1.1, CONSTRAINTS['u_max'] * 1.1])

    axes[3].plot(log['t'], log['cost'])
    axes[3].set_ylabel('MPC cost')
    axes[3].set_xlabel('Time [s]')
    axes[3].set_yscale('log')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120)
    return fig


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tau', type=int, default=3, choices=[0, 1, 2, 3])
    parser.add_argument('--time', type=float, default=8.0)
    parser.add_argument('--out', type=str, default='./nmpc_test.png')
    args = parser.parse_args()

    print(f"[INFO] Test NMPC -> EP{args.tau}")
    log = run_nmpc_only(target_tau=args.tau, total_time=args.time)
    plot_nmpc_run(log, args.tau, save_path=args.out)
    plt.show()
    print(f"[INFO] Plot: {args.out}")
