"""
Script di valutazione con animazione integrata del doppio pendolo.

Supporta tre modalità di controller:
  --controller nmpc      → usa solo NMPC (nessun modello RL necessario)
  --controller rl        → usa solo RL (richiede --model)
  --controller hybrid    → usa architettura ibrida (richiede --model + --mode)

E tre scenari di test:
  --scenario stabilize   → stabilizzazione su un singolo EP
  --scenario transitions → sequenza dei 12 scenari di transizione (Lee et al.)
  --scenario recovery    → test di recovery con disturbo esterno

Esempi d'uso immediato (senza modello RL addestrato):
  python -m training.evaluate_animate --controller nmpc --scenario stabilize --tau 3
  python -m training.evaluate_animate --controller nmpc --scenario transitions
  python -m training.evaluate_animate --controller nmpc --scenario recovery --tau 3

Con modello RL:
  python -m training.evaluate_animate --controller rl --model models_saved/tqc_safety_filter/final.zip --scenario transitions
"""

import sys
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from pathlib import Path

from controllers.nmpc import NMPCController
from models.dynamics import rk4_step
from configs.config import (
    EQUILIBRIUM_POINTS, SIM_PARAMS, CONSTRAINTS, PHYSICAL_PARAMS as P
)

# Sequenza 12 transizioni (Fig. 11, Lee et al.)
TRANSITION_SEQUENCE = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (0, 2), (2, 0), (0, 3), (3, 2),
    (2, 1), (1, 3), (3, 1), (1, 0),
]


# ──────────────────────────────────────────────────────────────
# Controller wrappers
# ──────────────────────────────────────────────────────────────

class NMPCAgent:
    """Wrapper per usare NMPC come agente di controllo."""
    def __init__(self):
        print("[INFO] Inizializzazione NMPC controller...")
        self.mpc = NMPCController(verbose=False)
        self._tau = 3

    def set_target(self, tau):
        self._tau = tau
        self.mpc.reset()

    def predict(self, state):
        try:
            u, _, _, _ = self.mpc.solve(state, self._tau)
        except Exception as e:
            print(f"[WARN] NMPC solver fallito: {e}")
            u = 0.0
        return u


class RLAgent:
    """Wrapper per usare un modello TQC addestrato."""
    def __init__(self, model_path, mode='vanilla', mpc=None):
        from sb3_contrib import TQC
        print(f"[INFO] Caricamento modello RL: {model_path}")
        self.model = TQC.load(model_path)
        self.mpc = mpc
        self.mode = mode
        self._tau = 3
        self._u_max = CONSTRAINTS['u_max']

    def set_target(self, tau):
        self._tau = tau
        if self.mpc:
            self.mpc.reset()

    def _build_obs(self, state):
        y, th1, th2, ydot, th1d, th2d = state
        return np.array([
            y, np.sin(th1), np.cos(th1),
            np.sin(th2), np.cos(th2),
            ydot, th1d, th2d, float(self._tau)
        ], dtype=np.float32)

    def predict(self, state):
        obs = self._build_obs(state)
        action, _ = self.model.predict(obs, deterministic=True)
        a = np.asarray(action, dtype=np.float32).item()
        u_rl = float(np.clip(a, -1.0, 1.0) * self._u_max)

        if self.mode == 'safety_filter' and self.mpc:
            y, _, _, ydot, _, _ = state
            y_warn = 0.85 * CONSTRAINTS['y_max']
            if ((y > y_warn and ydot > 0 and u_rl > 0) or
                    (y < -y_warn and ydot < 0 and u_rl < 0)):
                try:
                    u_mpc, _, _, _ = self.mpc.solve(state, self._tau)
                    return u_mpc
                except Exception:
                    pass
        elif self.mode == 'hierarchical' and self.mpc:
            try:
                u_mpc, _, _, _ = self.mpc.solve(state, self._tau)
                delta = a * (self._u_max / 3.0)
                return float(np.clip(u_mpc + delta,
                                     CONSTRAINTS['u_min'],
                                     CONSTRAINTS['u_max']))
            except Exception:
                pass
        return u_rl


# ──────────────────────────────────────────────────────────────
# Simulation routines
# ──────────────────────────────────────────────────────────────

def wrap_angle(x):
    return (x + np.pi) % (2*np.pi) - np.pi

def _simulate(agent, state_init, tau_sequence, time_per_segment):
    """Core loop: simula una lista di segmenti con cambio di target."""
    dt_ode = SIM_PARAMS['dt']
    dt_ctrl = SIM_PARAMS['control_dt']
    n_sub = int(dt_ctrl / dt_ode)
    n_steps = int(time_per_segment / dt_ctrl)

    log = {'t': [], 'y': [], 'th1': [], 'th2': [],
           'ydot': [], 'th1d': [], 'th2d': [],
           'u': [], 'tau': [], 'segment': []}
    state = state_init.copy()
    t = 0.0

    for seg_idx, tau_target in enumerate(tau_sequence):
        agent.set_target(tau_target)
        for _ in range(n_steps):
            u = agent.predict(state)
            u = float(np.clip(u, CONSTRAINTS['u_min'], CONSTRAINTS['u_max']))

            log['t'].append(t)
            log['y'].append(state[0])
            log['th1'].append(wrap_angle(state[1]))
            log['th2'].append(wrap_angle(state[2]))
            log['ydot'].append(state[3])
            log['th1d'].append(state[4])
            log['th2d'].append(state[5])
            log['u'].append(u)
            log['tau'].append(tau_target)
            log['segment'].append(seg_idx)

            for _ in range(n_sub):
                state = rk4_step(state, u, dt_ode)
            t += dt_ctrl

            if abs(state[0]) > CONSTRAINTS['y_max'] * 1.05:
                print(f"[WARN] Limite rail superato a t={t:.2f}s, "
                      f"tau={tau_target}")
                state[0] = np.clip(state[0], -CONSTRAINTS['y_max'],
                                   CONSTRAINTS['y_max'])
                state[3] = 0.0

    return {k: np.array(v) for k, v in log.items()}


def run_stabilization(agent, tau, total_time=5.0, perturb=0.05):
    ep = EQUILIBRIUM_POINTS[tau]
    x0 = np.array([0.0, ep['theta1'] + perturb,
                    ep['theta2'] + perturb * 0.5, 0.0, 0.0, 0.0])
    print(f"[SIM] Stabilizzazione -> EP{tau} ({ep['name']})")
    return _simulate(agent, x0, [tau], total_time)


def run_transitions(agent, time_per=6.0):
    x0 = np.array([0.0, -np.pi, 0.0, 0.0, 0.0, 0.0])
    tau_sequence = [tau_to for (_, tau_to) in TRANSITION_SEQUENCE]
    print(f"[SIM] 12 transizioni: {tau_sequence}")
    return _simulate(agent, x0, tau_sequence, time_per)


def run_recovery(agent, tau=3, total_time=12.0, disturb_at=5.0,
                 disturb_mag=8.0):
    ep = EQUILIBRIUM_POINTS[tau]
    # Parti da EP0 e fai prima swing-up
    x0 = np.array([0.0, -np.pi, 0.0, 0.0, 0.0, 0.0])
    agent.set_target(tau)

    dt_ode = SIM_PARAMS['dt']
    dt_ctrl = SIM_PARAMS['control_dt']
    n_sub = int(dt_ctrl / dt_ode)
    n_steps = int(total_time / dt_ctrl)
    disturb_step = int(disturb_at / dt_ctrl)

    log = {'t': [], 'y': [], 'th1': [], 'th2': [],
           'ydot': [], 'th1d': [], 'th2d': [],
           'u': [], 'tau': [], 'disturb': []}
    state = x0.copy()
    t = 0.0

    for k in range(n_steps):
        u = agent.predict(state)
        u = float(np.clip(u, CONSTRAINTS['u_min'], CONSTRAINTS['u_max']))

        disturbed = (k == disturb_step)
        if disturbed:
            state[4] += disturb_mag
            state[5] -= disturb_mag
            print(f"[SIM] DISTURBO applicato a t={t:.2f}s "
                  f"(dth1+={disturb_mag}, dth2-={disturb_mag})")

        log['t'].append(t)
        log['y'].append(state[0])
        log['th1'].append(wrap_angle(state[1]))
        log['th2'].append(wrap_angle(state[2]))
        log['ydot'].append(state[3])
        log['th1d'].append(state[4])
        log['th2d'].append(state[5])
        log['u'].append(u)
        log['tau'].append(tau)
        log['disturb'].append(disturbed)

        for _ in range(n_sub):
            state = rk4_step(state, u, dt_ode)
        t += dt_ctrl

        if abs(state[0]) > CONSTRAINTS['y_max'] * 1.05:
            state[0] = np.clip(state[0], -CONSTRAINTS['y_max'],
                               CONSTRAINTS['y_max'])
            state[3] = 0.0

    return {k: np.array(v) for k, v in log.items()}


# ──────────────────────────────────────────────────────────────
# Kinematics
# ──────────────────────────────────────────────────────────────

def fk(y, th1, th2):
    cart = np.array([y, 0.0])
    j1 = cart
    j2 = j1 + np.array([np.sin(th1) * P['L1'], np.cos(th1) * P['L1']])
    L2v = 2 * P['l2']
    tip = j2 + np.array([np.sin(th1 + th2) * L2v,
                          np.cos(th1 + th2) * L2v])
    return cart, j1, j2, tip


# ──────────────────────────────────────────────────────────────
# Animation + plots
# ──────────────────────────────────────────────────────────────

def build_figure(log, title=''):
    t = log['t']
    y = log['y']
    th1 = log['th1']
    th2 = log['th2']
    u = log['u']
    tau = log['tau']
    disturb = log.get('disturb', np.zeros(len(t), dtype=bool))

    fig = plt.figure(figsize=(9, 13))
    fig.patch.set_facecolor('#1a1a2e')

    # Pendolo in alto; tutti i plot temporali impilati sotto.
    # La prima riga è più alta per dare spazio all'animazione.
    # gs = fig.add_gridspec(
    #     6, 1,
    #     height_ratios=[4.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    #     hspace=0.3,
    #     left=0.09, right=0.98,
    #     top=0.99, bottom=0.03
    # )


    gs_main = fig.add_gridspec(
        2, 1,
        height_ratios=[4.0, 1.0],
        hspace=0.1,
        left=0.12,
        right=0.98,
        top=0.99,
        bottom=0.05
    )

    ax_p = fig.add_subplot(gs_main[0, 0])

    # Layout secondario:
    # riduce soltanto lo spazio tra i plot temporali
    # gs_plots = gs_main[1, 0].subgridspec(
    #     4, 1,
    #     hspace=0.15
    # )


    # ax_p = fig.add_subplot(gs_plots[0, 0])
    # ax_th1 = fig.add_subplot(gs_plots[0, 0])
    # ax_th2 = fig.add_subplot(gs_plots[1, 0], sharex=ax_th1)
    # ax_y = fig.add_subplot(gs_plots[2, 0], sharex=ax_th1)
    ax_u = fig.add_subplot(gs_main[1, 0])
    # ax_ep = fig.add_subplot(gs_plots[4, 0], sharex=ax_th1)

    # Aggiornata la lista degli assi per l'applicazione dello stile
    for ax in [ax_p, ax_u]: # ax_th1, ax_th2, ax_y, ax_u, ax_ep]:
        ax.set_facecolor('#16213e')
        for spine in ax.spines.values():
            spine.set_color('#0f3460')
        ax.tick_params(colors='#e0e0e0', labelsize=8)
        ax.yaxis.label.set_color('#e0e0e0')
        ax.xaxis.label.set_color('#e0e0e0')
        ax.grid(color='#0f3460', alpha=0.6, linewidth=0.5)

    if title:
        fig.suptitle(title, color='#e0e0e0', fontsize=13, fontweight='bold')

    # ── pendulum panel ──
    pad = 0.06
    ax_p.set_xlim(-CONSTRAINTS['y_max'] - pad,
                   CONSTRAINTS['y_max'] + pad)
    ax_p.set_ylim(-0.50, 0.55)
    ax_p.set_aspect('equal')
    ax_p.set_xlabel('y  [m]', color='#e0e0e0')
    ax_p.set_ylabel('z  [m]', color='#e0e0e0')

    # Rail
    ax_p.plot([-CONSTRAINTS['y_max'], CONSTRAINTS['y_max']], [0, 0],
              color='#4a4e69', lw=3)
    for xw in [-CONSTRAINTS['y_max'], CONSTRAINTS['y_max']]:
        ax_p.plot([xw, xw], [-0.05, 0.05], color='#9a8c98', lw=5)

    cw, ch = 0.07, 0.04
    cart_patch = Rectangle((-cw / 2, -ch / 2), cw, ch,
                            facecolor='#4cc9f0', edgecolor='#4895ef',
                            linewidth=1.5, zorder=5)
    ax_p.add_patch(cart_patch)

    PEND1_COLOR = '#f72585'
    PEND2_COLOR = '#4cc9f0'
    TRACE_COLOR = '#f8961e'
    

    p1_line, = ax_p.plot([], [], '-', color=PEND1_COLOR, lw=5, zorder=6)
    p2_line, = ax_p.plot([], [], '-', color=PEND2_COLOR, lw=4, zorder=6)
    j1_dot,  = ax_p.plot([], [], 'o', color='#ffffff', ms=7, zorder=7)
    j2_dot,  = ax_p.plot([], [], 'o', color='#ffffff', ms=6, zorder=7)
    tip_dot, = ax_p.plot([], [], 'o', color=TRACE_COLOR, ms=9, zorder=8)
    trace_line, = ax_p.plot([], [], '-', color=TRACE_COLOR,
                              lw=1.0, alpha=0.4, zorder=4)
    trace_x, trace_y_arr = [], []

    time_txt = ax_p.text(
        0.02, 0.97, '', transform=ax_p.transAxes,
        fontsize=10, fontweight='bold', color='#e0e0e0',
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3',
                  facecolor='#0f3460', edgecolor='#4cc9f0', alpha=0.9)
    )
    distmark_line = ax_p.axvline(x=0, color='#f72585', lw=2,
                                  ls='--', alpha=0, zorder=9)

    # EP target ghost
    ep_ghost_p1, = ax_p.plot([], [], '--', color=PEND1_COLOR,
                               alpha=0.25, lw=2)
    ep_ghost_p2, = ax_p.plot([], [], '--', color=PEND2_COLOR,
                               alpha=0.25, lw=2)

    # ── time series ──
    
    # # Subplot per Theta 1
    # ax_th1.plot(t, th1, color=PEND1_COLOR, lw=0.8, alpha=0.5, label=r'$\theta_1$')
    # th1_cursor, = ax_th1.plot([], [], 'o', color=PEND1_COLOR, ms=5)
    # ax_th1.legend(loc='upper right', fontsize=8, labelcolor='#e0e0e0', facecolor='#16213e')
    # ax_th1.set_ylabel(r'$\theta_1$ [rad]')
    
    # # Subplot per Theta 2
    # ax_th2.plot(t, th2, color=PEND2_COLOR, lw=0.8, alpha=0.5, label=r'$\theta_2$')
    # th2_cursor, = ax_th2.plot([], [], 'o', color=PEND2_COLOR, ms=5)
    # ax_th2.legend(loc='upper right', fontsize=8, labelcolor='#e0e0e0', facecolor='#16213e')
    # ax_th2.set_ylabel(r'$\theta_2$ [rad]')

    # # Subplot per Cart (y)
    # ax_y.plot(t, y, color='#80ffdb', lw=0.8, alpha=0.5)
    # ax_y.axhline(+CONSTRAINTS['y_max'], color='#f72585', ls='--', lw=0.8, alpha=0.6)
    # ax_y.axhline(-CONSTRAINTS['y_max'], color='#f72585', ls='--', lw=0.8, alpha=0.6)
    # y_cursor, = ax_y.plot([], [], 'o', color='#80ffdb', ms=5)
    # ax_y.set_ylabel('Cart  [m]')

    # Subplot per Control Action (u)
    ax_u.plot(t, u, color='#ffb347', lw=0.8, alpha=0.5)
    ax_u.axhline(+CONSTRAINTS['u_max'], color='#ff6b6b', ls='--', lw=0.8, alpha=0.5)
    ax_u.axhline(-CONSTRAINTS['u_max'], color='#ff6b6b', ls='--', lw=0.8, alpha=0.5)
    u_cursor, = ax_u.plot([], [], 'o', color='#ffb347', ms=5, zorder=5)
    ax_u.set_ylabel(r'u  [m/s²]')
    ax_u.set_xlabel('Time  [s]')

    # Subplot per Target EP
    tau_arr = log['tau'].astype(int) if hasattr(log['tau'], 'astype') else np.array(log['tau'], dtype=int)
    # ep_colors = ['#9b5de5', '#f15bb5', '#fee440', '#00bbf9']
    # ep_names  = ['EP0\n(D-D)', 'EP1\n(D-U)', 'EP2\n(U-D)', 'EP3\n(U-U)']
    # for ep_i in range(4):
    #     mask = tau_arr == ep_i
    #     if mask.any():
    #         ax_ep.scatter(np.array(t)[mask], np.ones(mask.sum()) * ep_i,
    #                       c=ep_colors[ep_i], s=2, alpha=0.7)
    # ep_cursor, = ax_ep.plot([], [], 's', color='white', ms=6, zorder=5)
    # ax_ep.set_yticks([0, 1, 2, 3])
    # ax_ep.set_yticklabels(ep_names, fontsize=7)
    # ax_ep.set_ylabel('Target EP')
    # ax_ep.set_xlabel('Time  [s]')
    # ax_ep.set_ylim(-0.5, 3.5)

    # Mostra le etichette temporali solo nell'ultimo subplot.
    for ax in [ax_u]: # ax_th1, ax_th2, ax_y, ax_u]:
        ax.tick_params(labelbottom=True)

    return (fig, ax_p,
            cart_patch, p1_line, p2_line, j1_dot, j2_dot, tip_dot,
            trace_line, trace_x, trace_y_arr,
            time_txt, distmark_line,
            ep_ghost_p1, ep_ghost_p2, u_cursor,
            # th1_cursor, th2_cursor, y_cursor, u_cursor, ep_cursor,
            t, y, th1, th2, u, tau_arr, disturb)


def make_animation(log, title='', fps=30, speed=1.0, save_path=None):
    """Costruisce e ritorna (ani, fig)."""

    (fig, ax_p,
     cart_patch, p1_line, p2_line, j1_dot, j2_dot, tip_dot,
     trace_line, trace_x, trace_y_arr,
     time_txt, distmark_line,
     ep_ghost_p1, ep_ghost_p2, u_cursor,
     # th1_cursor, th2_cursor, y_cursor, u_cursor, ep_cursor,
     t, y, th1, th2, u, tau_arr, disturb) = build_figure(log, title)

    dt_data = t[1] - t[0] if len(t) > 1 else 0.01
    stride  = max(1, int(round((1.0 / fps / speed) / dt_data)))
    idx_all = np.arange(0, len(t), stride)
    n_frames = len(idx_all)

    L2v = 2 * P['l2']

    def update(fi):
        i = idx_all[fi]
        cart_pos, j1, j2, tip = fk(y[i], th1[i], th2[i])

        # Cart
        cw, ch = 0.07, 0.04
        cart_patch.set_xy((cart_pos[0] - cw / 2, cart_pos[1] - ch / 2))

        # Pendoli
        p1_line.set_data([j1[0], j2[0]], [j1[1], j2[1]])
        p2_line.set_data([j2[0], tip[0]], [j2[1], tip[1]])
        j1_dot.set_data([j1[0]], [j1[1]])
        j2_dot.set_data([j2[0]], [j2[1]])
        tip_dot.set_data([tip[0]], [tip[1]])

        # Trace (ultimi 150 punti)
        trace_x.append(tip[0])
        trace_y_arr.append(tip[1])
        if len(trace_x) > 150:
            trace_x.pop(0)
            trace_y_arr.pop(0)
        trace_line.set_data(trace_x, trace_y_arr)

        # Ghost target
        tau_i = int(tau_arr[i])
        ep = EQUILIBRIUM_POINTS[tau_i]
        gj2 = np.array([0.0, 0.0]) + np.array(
            [np.sin(ep['theta1']) * P['L1'],
             np.cos(ep['theta1']) * P['L1']])
        th2_abs_t = ep['theta1'] + ep['theta2']
        gtip = gj2 + np.array([np.sin(th2_abs_t) * L2v,
                                 np.cos(th2_abs_t) * L2v])
        ep_ghost_p1.set_data([0, gj2[0]], [0, gj2[1]])
        ep_ghost_p2.set_data([gj2[0], gtip[0]], [gj2[1], gtip[1]])

        # Disturbo marker
        if disturb[i]:
            distmark_line.set_alpha(0.8)
            distmark_line.set_xdata([cart_pos[0]])
        else:
            distmark_line.set_alpha(0.0)

        # Testo tempo
        ep_name = EQUILIBRIUM_POINTS[tau_i]['name']
        time_txt.set_text(f' t = {t[i]:.2f} s   target: {ep_name} ')

        # Cursori
        # th1_cursor.set_data([t[i]], [th1[i]])
        # th2_cursor.set_data([t[i]], [th2[i]])
        # y_cursor.set_data([t[i]],   [y[i]])
        u_cursor.set_data([t[i]],   [u[i]])
        # ep_cursor.set_data([t[i]],  [tau_arr[i]])

        return (cart_patch, p1_line, p2_line,
                j1_dot, j2_dot, tip_dot, trace_line,
                ep_ghost_p1, ep_ghost_p2,
                distmark_line, time_txt, u_cursor)
                # th1_cursor, th2_cursor, y_cursor, u_cursor, ep_cursor)

    ani = animation.FuncAnimation(
        fig, update, frames=n_frames,
        interval=1000.0 / fps,
        blit=False, repeat=True
    )

    if save_path:
        save_path = str(save_path)
        print(f"[INFO] Salvataggio: {save_path}")
        if save_path.endswith('.gif'):
            ani.save(save_path, writer='pillow', fps=fps, dpi=90)
        elif save_path.endswith('.mp4'):
            ani.save(save_path, writer='ffmpeg', fps=fps, dpi=120, bitrate=2000)
        else:
            raise ValueError("Estensione non supportata (.gif o .mp4)")
        print(f"[INFO] Animazione salvata in {save_path}")

    return ani, fig


# ──────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluation + animazione doppio pendolo'
    )
    parser.add_argument('--controller',
                        choices=['nmpc', 'rl', 'hybrid'],
                        default='nmpc',
                        help='Tipo di controller da usare')
    parser.add_argument('--scenario',
                        choices=['stabilize', 'transitions', 'recovery'],
                        default='stabilize',
                        help='Scenario di test')
    parser.add_argument('--tau', type=int, default=3, choices=[0,1,2,3],
                        help='Target EP per stabilize/recovery')
    parser.add_argument('--model', type=str, default=None,
                        help='Path modello RL (.zip)')
    parser.add_argument('--mode',
                        choices=['vanilla','reward_shaper',
                                 'safety_filter','hierarchical'],
                        default='vanilla',
                        help='Modalità hybrid')
    parser.add_argument('--time', type=float, default=5.0,
                        help='Tempo per transizione/segmento (s)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frame per secondo animazione')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Velocità playback (1.0=realtime, 2.0=doppio)')
    parser.add_argument('--out', type=str, default=None,
                        help='Salva animazione (.gif o .mp4)')
    parser.add_argument('--no-show', action='store_true',
                        help='Non mostrare la finestra interattiva')
    args = parser.parse_args()

    # ── build agent ──
    if args.controller == 'nmpc':
        agent = NMPCAgent()
        ctrl_label = 'NMPC'
    elif args.controller == 'rl':
        if not args.model:
            print('[ERR] --controller rl richiede --model <path>')
            sys.exit(1)
        agent = RLAgent(args.model)
        ctrl_label = f'RL ({args.mode})'
    else:  # hybrid
        if not args.model:
            print('[ERR] --controller hybrid richiede --model <path>')
            sys.exit(1)
        mpc = NMPCController(verbose=False)
        agent = RLAgent(args.model, mode=args.mode, mpc=mpc)
        ctrl_label = f'Hybrid {args.mode}'

    # ── run scenario ──
    if args.scenario == 'stabilize':
        log = run_stabilization(agent, tau=args.tau,
                                 total_time=args.time,
                                 perturb=0.05)
        ep_name = EQUILIBRIUM_POINTS[args.tau]['name']
        title = f'{ctrl_label}  |  Stabilization → {ep_name}'

    elif args.scenario == 'transitions':
        log = run_transitions(agent, time_per=args.time)
        # title = f'{ctrl_label}  |  12 Transition Scenarios (Lee et al. 2025)'

    else:  # recovery
        log = run_recovery(agent, tau=args.tau, total_time=args.time * 3)
        ep_name = EQUILIBRIUM_POINTS[args.tau]['name']
        title = f'{ctrl_label}  |  Recovery Test → {ep_name}'

    print(f'\n[RESULT] Passi simulati: {len(log["t"])}')
    print(f'[RESULT] Durata:         {log["t"][-1]:.2f} s')
    print(f'[RESULT] y finale:       {log["y"][-1]:+.4f} m')
    print(f'[RESULT] th1 finale:     {log["th1"][-1]:+.4f} rad')
    print(f'[RESULT] th2 finale:     {log["th2"][-1]:+.4f} rad')

    # ── animate ──
    if not args.no_show:
        matplotlib.use('TkAgg' if args.out is None else 'Agg')

    ani, fig = make_animation(log, # title=title,
                               fps=args.fps, speed=args.speed,
                               save_path=args.out)
    if not args.no_show:
        plt.show()
