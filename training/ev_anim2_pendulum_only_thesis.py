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
# Animazione del solo doppio pendolo
# ──────────────────────────────────────────────────────────────

# Palette chiara e sobria, adatta a figure e animazioni per una tesi.
FIG_BG = '#FFFFFF'
AX_BG = '#FBFCFE'
TEXT_COLOR = '#263238'
SPINE_COLOR = '#AEB8C4'
GRID_COLOR = '#D9E0E8'
RAIL_COLOR = '#4B5563'
ENDSTOP_COLOR = '#7B8794'
CART_FACE = '#DCEAF7'
CART_EDGE = '#1F4E79'
PEND1_COLOR = '#1F4E79'       # blu accademico
PEND2_COLOR = '#B45F06'       # arancio bruciato
TRACE_COLOR = '#5F6B76'
JOINT_FACE = '#FFFFFF'
JOINT_EDGE = '#374151'
DISTURB_COLOR = '#9B2C2C'
GHOST_COLOR = '#9AA6B2'


def build_figure(log, title=''):
    """Crea una figura contenente esclusivamente il doppio pendolo."""
    t = np.asarray(log['t'])
    y = np.asarray(log['y'])
    th1 = np.asarray(log['th1'])
    th2 = np.asarray(log['th2'])
    tau_arr = np.asarray(log['tau'], dtype=int)
    disturb = np.asarray(
        log.get('disturb', np.zeros(len(t), dtype=bool)),
        dtype=bool,
    )

    fig, ax_p = plt.subplots(figsize=(9.2, 6.2))
    fig.patch.set_facecolor(FIG_BG)
    ax_p.set_facecolor(AX_BG)
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.12, top=0.95)

    # Titolo opzionale: nello script principale non viene passato, quindi
    # l'animazione resta pulita e mostra soltanto il pendolo.
    if title:
        ax_p.set_title(title, fontsize=12, color=TEXT_COLOR, pad=12)

    pad_x = 0.08
    total_length = P['L1'] + 2 * P['l2']
    vertical_pad = 0.08
    ax_p.set_xlim(
        -CONSTRAINTS['y_max'] - pad_x,
        CONSTRAINTS['y_max'] + pad_x,
    )
    ax_p.set_ylim(-total_length - vertical_pad,
                  total_length + vertical_pad)
    ax_p.set_aspect('equal', adjustable='box')
    ax_p.set_xlabel(r'$y$ [m]',
                    fontsize=14, color=TEXT_COLOR)
    ax_p.set_ylabel(r'$z$ [m]',
                    fontsize=14, color=TEXT_COLOR)

    ax_p.grid(True, color=GRID_COLOR, linewidth=0.7, alpha=0.75)
    ax_p.tick_params(colors=TEXT_COLOR, labelsize=14)
    for spine in ax_p.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(0.9)

    # Rotaia e fine corsa.
    ax_p.plot(
        [-CONSTRAINTS['y_max'], CONSTRAINTS['y_max']],
        [0.0, 0.0],
        color=RAIL_COLOR,
        lw=2.8,
        solid_capstyle='round',
        zorder=2,
    )
    for x_stop in (-CONSTRAINTS['y_max'], CONSTRAINTS['y_max']):
        ax_p.plot(
            [x_stop, x_stop], [-0.045, 0.045],
            color=ENDSTOP_COLOR,
            lw=4.0,
            solid_capstyle='round',
            zorder=2,
        )

    # Carrello.
    cart_width, cart_height = 0.07, 0.04
    cart_patch = Rectangle(
        (-cart_width / 2, -cart_height / 2),
        cart_width,
        cart_height,
        facecolor=CART_FACE,
        edgecolor=CART_EDGE,
        linewidth=1.5,
        zorder=5,
    )
    ax_p.add_patch(cart_patch)

    # Aste, giunti e traccia della punta.
    p1_line, = ax_p.plot(
        [], [], '-', color=PEND1_COLOR, lw=4.2,
        solid_capstyle='round', zorder=6,
    )
    p2_line, = ax_p.plot(
        [], [], '-', color=PEND2_COLOR, lw=3.8,
        solid_capstyle='round', zorder=6,
    )
    j1_dot, = ax_p.plot(
        [], [], 'o', ms=7.0,
        markerfacecolor=JOINT_FACE,
        markeredgecolor=JOINT_EDGE,
        markeredgewidth=1.2,
        zorder=7,
    )
    j2_dot, = ax_p.plot(
        [], [], 'o', ms=6.5,
        markerfacecolor=JOINT_FACE,
        markeredgecolor=JOINT_EDGE,
        markeredgewidth=1.2,
        zorder=7,
    )
    tip_dot, = ax_p.plot(
        [], [], 'o', ms=7.5,
        markerfacecolor=PEND2_COLOR,
        markeredgecolor=JOINT_FACE,
        markeredgewidth=1.0,
        zorder=8,
    )
    trace_line, = ax_p.plot(
        [], [], '-', color=TRACE_COLOR,
        lw=1.0, alpha=0.45, zorder=3,
    )
    trace_x, trace_z = [], []

    # Configurazione di equilibrio richiesta, mostrata in modo discreto.
    ep_ghost_p1, = ax_p.plot(
        [], [], '--', color=GHOST_COLOR,
        alpha=0.65, lw=1.4, zorder=4,
    )
    ep_ghost_p2, = ax_p.plot(
        [], [], '--', color=GHOST_COLOR,
        alpha=0.65, lw=1.4, zorder=4,
    )

    time_txt = ax_p.text(
        0.025, 0.97, '',
        transform=ax_p.transAxes,
        fontsize=11,
        color=TEXT_COLOR,
        fontweight='bold',
        verticalalignment='top',
        bbox=dict(
            boxstyle='round,pad=0.30',
            facecolor='#FFFFFF',
            edgecolor=SPINE_COLOR,
            linewidth=0.8,
            alpha=0.95,
        ),
        zorder=10,
    )

    distmark_line = ax_p.axvline(
        x=0.0,
        color=DISTURB_COLOR,
        lw=1.6,
        ls='--',
        alpha=0.0,
        zorder=9,
    )

    return (
        fig, ax_p,
        cart_patch, p1_line, p2_line,
        j1_dot, j2_dot, tip_dot,
        trace_line, trace_x, trace_z,
        time_txt, distmark_line,
        ep_ghost_p1, ep_ghost_p2,
        t, y, th1, th2, tau_arr, disturb,
    )


def make_animation(log, title='', fps=30, speed=1.0, save_path=None):
    """Costruisce l'animazione del solo pendolo e ritorna ``(ani, fig)``."""
    (
        fig, ax_p,
        cart_patch, p1_line, p2_line,
        j1_dot, j2_dot, tip_dot,
        trace_line, trace_x, trace_z,
        time_txt, distmark_line,
        ep_ghost_p1, ep_ghost_p2,
        t, y, th1, th2, tau_arr, disturb,
    ) = build_figure(log, title)

    if len(t) == 0:
        raise ValueError('Il log di simulazione è vuoto.')
    if fps <= 0:
        raise ValueError('--fps deve essere maggiore di zero.')
    if speed <= 0:
        raise ValueError('--speed deve essere maggiore di zero.')

    dt_data = t[1] - t[0] if len(t) > 1 else 0.01
    stride = max(1, int(round((1.0 / fps / speed) / dt_data)))
    idx_all = np.arange(0, len(t), stride)
    n_frames = len(idx_all)

    second_link_length = 2 * P['l2']
    cart_width, cart_height = 0.07, 0.04

    def update(frame_index):
        i = idx_all[frame_index]
        cart_pos, j1, j2, tip = fk(y[i], th1[i], th2[i])

        cart_patch.set_xy((
            cart_pos[0] - cart_width / 2,
            cart_pos[1] - cart_height / 2,
        ))

        p1_line.set_data([j1[0], j2[0]], [j1[1], j2[1]])
        p2_line.set_data([j2[0], tip[0]], [j2[1], tip[1]])
        j1_dot.set_data([j1[0]], [j1[1]])
        j2_dot.set_data([j2[0]], [j2[1]])
        tip_dot.set_data([tip[0]], [tip[1]])

        # Traccia limitata agli ultimi 150 campioni visualizzati.
        trace_x.append(tip[0])
        trace_z.append(tip[1])
        if len(trace_x) > 150:
            trace_x.pop(0)
            trace_z.pop(0)
        trace_line.set_data(trace_x, trace_z)

        # La configurazione obiettivo viene riferita alla posizione attuale
        # del carrello, così resta geometricamente coerente durante il moto.
        tau_i = int(tau_arr[i])
        equilibrium = EQUILIBRIUM_POINTS[tau_i]
        ghost_j1 = cart_pos
        ghost_j2 = ghost_j1 + np.array([
            np.sin(equilibrium['theta1']) * P['L1'],
            np.cos(equilibrium['theta1']) * P['L1'],
        ])
        ghost_theta2_abs = (
            equilibrium['theta1'] + equilibrium['theta2']
        )
        ghost_tip = ghost_j2 + np.array([
            np.sin(ghost_theta2_abs) * second_link_length,
            np.cos(ghost_theta2_abs) * second_link_length,
        ])
        ep_ghost_p1.set_data(
            [ghost_j1[0], ghost_j2[0]],
            [ghost_j1[1], ghost_j2[1]],
        )
        ep_ghost_p2.set_data(
            [ghost_j2[0], ghost_tip[0]],
            [ghost_j2[1], ghost_tip[1]],
        )

        if disturb[i]:
            distmark_line.set_alpha(0.75)
            distmark_line.set_xdata([cart_pos[0], cart_pos[0]])
        else:
            distmark_line.set_alpha(0.0)

        ep_name = equilibrium['name']
        time_txt.set_text(f't = {t[i]:.2f} s   |   target: {ep_name}')

        return (
            cart_patch, p1_line, p2_line,
            j1_dot, j2_dot, tip_dot,
            trace_line, ep_ghost_p1, ep_ghost_p2,
            distmark_line, time_txt,
        )

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=n_frames,
        interval=1000.0 / fps,
        blit=False,
        repeat=True,
    )

    if save_path:
        save_path = str(save_path)
        print(f'[INFO] Salvataggio: {save_path}')
        if save_path.lower().endswith('.gif'):
            ani.save(
                save_path,
                writer='pillow',
                fps=fps,
                dpi=120,
                savefig_kwargs={'facecolor': FIG_BG},
            )
        elif save_path.lower().endswith('.mp4'):
            ani.save(
                save_path,
                writer='ffmpeg',
                fps=fps,
                dpi=150,
                bitrate=2400,
                savefig_kwargs={'facecolor': FIG_BG},
            )
        else:
            raise ValueError('Estensione non supportata: usare .gif o .mp4')
        print(f'[INFO] Animazione salvata in {save_path}')

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
    parser.add_argument('--time', type=float, default=8.0,
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
