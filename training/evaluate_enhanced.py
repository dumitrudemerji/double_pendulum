"""Evaluation of RL controllers for double-pendulum transition and recovery tests.

Examples
--------
python -m training.evaluate_enhanced \\
    --model models_saved/tqc_safety_filter/final.zip \\
    --mode safety_filter

python -m training.evaluate_enhanced \\
    --model models_saved/tqc_safety_filter/final.zip \\
    --mode safety_filter --disturbance --format pdf --no-show
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from sb3_contrib import TQC
from controllers.environment import DoublePendulumEnv
from controllers.nmpc import NMPCController
from configs.config import EQUILIBRIUM_POINTS, SIM_PARAMS, CONSTRAINTS


# Sequence of 12 transition scenarios (Fig. 11 of Lee et al. 2025)
TRANSITION_SEQUENCE = [
    (0, 1), (1, 2), (2, 3), (3, 0),
    (0, 2), (2, 0), (0, 3), (3, 2),
    (2, 1), (1, 3), (3, 1), (1, 0),
]

# TRANSITION_SEQUENCE = [(1,3)]

# A restrained, print-friendly palette. SVG/PDF remain vector graphics.
COLORS = {
    "cart": "#1F77B4",
    "theta1": "#D55E00",
    "theta2": "#009E73",
    "reference": "#555555",
    "input": "#4C4C4C",
    "target": "#7B2CBF",
    "constraint": "#B2182B",
    "disturbance": "#C1121F",
    "grid": "#D0D5DD",
    "shade": "#EEF2F6",
}


def configure_plot_style():
    """Set a consistent style suitable for papers, reports, and slides."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 17,
        "axes.titlesize": 17,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.titleweight": "bold",
        "axes.labelcolor": "#1F2937",
        "text.color": "#1F2937",
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "axes.edgecolor": "#9CA3AF",
        "lines.linewidth": 1.8,
        "lines.solid_capstyle": "round",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    })


def wrap_angle(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


def evaluate_transitions(model, mode, mpc=None, episode_duration=8.0):
    """Execute the sequence of 12 transitions and collect the trajectory."""
    env = DoublePendulumEnv(
        mode=mode, mpc_controller=mpc,
        randomize_tau=False, fixed_tau=0,
    )
    env.reset(seed=0)

    # Initialize at EP0 (down-down) with an exact state.
    env.state = np.array([0.0, -np.pi, 0.0, 0.0, 0.0, 0.0])

    log = {
        "t": [], "y": [], "th1": [], "th2": [],
        "ydot": [], "th1d": [], "th2d": [],
        "u": [], "tau": [], "reward": [],
    }

    dt = SIM_PARAMS["control_dt"]
    steps_per_transition = int(episode_duration / dt)
    t = 0.0

    for tau_from, tau_to in TRANSITION_SEQUENCE:
        env.tau = tau_to
        if mpc is not None:
            mpc.reset()

        for _ in range(steps_per_transition):
            obs = env._get_obs()
            action, _ = model.predict(obs, deterministic=True)
            _, reward, terminated, _, info = env.step(action)

            log["t"].append(t)
            log["y"].append(env.state[0])
            log["th1"].append(wrap_angle(env.state[1]))
            log["th2"].append(wrap_angle(env.state[2]))
            log["ydot"].append(env.state[3])
            log["th1d"].append(env.state[4])
            log["th2d"].append(env.state[5])
            log["u"].append(info["u_applied"])
            log["tau"].append(env.tau)
            log["reward"].append(reward)
            t += dt

            if terminated:
                print(
                    f"[WARN] Early termination at t={t:.2f}s "
                    f"(tau {tau_from}->{tau_to})"
                )
                ep = EQUILIBRIUM_POINTS[tau_to]
                env.state = np.array([
                    0.0, ep["theta1"], ep["theta2"], 0.0, 0.0, 0.0,
                ])

    return {key: np.asarray(value) for key, value in log.items()}


def evaluate_recovery(
    model,
    mode,
    mpc=None,
    target_tau=3,
    disturbance_time=5.0,
    disturbance_magnitude=15.0,
    total_time=10.0,
):
    """Evaluate recovery after an impulse in both angular velocities."""
    env = DoublePendulumEnv(
        mode=mode, mpc_controller=mpc,
        randomize_tau=False, fixed_tau=target_tau,
    )
    env.reset(seed=42)
    env.state = np.array([0.0, -np.pi, 0.0, 0.0, 0.0, 0.0])

    log = {"t": [], "y": [], "th1": [], "th2": [], "u": [], "disturbed": []}

    dt = SIM_PARAMS["control_dt"]
    n_steps = int(total_time / dt)
    disturbance_step = int(disturbance_time / dt)
    t = 0.0

    for step in range(n_steps):
        obs = env._get_obs()
        action, _ = model.predict(obs, deterministic=True)
        _, _, terminated, _, info = env.step(action)

        disturbed = False
        if step == disturbance_step:
            env.state[4] += disturbance_magnitude
            env.state[5] += disturbance_magnitude
            disturbed = True
            print(f"[INFO] Disturbance applied at t={t:.2f}s")

        log["t"].append(t)
        log["y"].append(env.state[0])
        log["th1"].append(wrap_angle(env.state[1]))
        log["th2"].append(wrap_angle(env.state[2]))
        log["u"].append(info["u_applied"])
        log["disturbed"].append(disturbed)
        t += dt

        if terminated:
            print(f"[WARN] Recovery failed at t={t:.2f}s")
            break

    return {key: np.asarray(value) for key, value in log.items()}


def _mode_label(mode_name):
    return mode_name.replace("_", " ").title()


def _setup_axis(ax, ylabel=None):
    """Apply the common visual treatment to a subplot."""
    ax.grid(True, which="major", axis="both", color=COLORS["grid"], linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.margins(x=0)
    if ylabel:
        ax.set_ylabel(ylabel)


def _transition_boundaries(log):
    """Return start/end indices for each target-equilibrium section."""
    tau = np.asarray(log["tau"])
    change_idx = np.flatnonzero(np.diff(tau) != 0) + 1
    starts = np.r_[0, change_idx]
    ends = np.r_[change_idx - 1, tau.size - 1]
    return starts, ends


def _shade_transition_sections(axes, log):
    """Use alternating background bands to make 12 transitions navigable."""
    starts, ends = _transition_boundaries(log)
    t = log["t"]

    for section, (start, end) in enumerate(zip(starts, ends), start=1):
        left = t[start]
        right = t[end] if end < len(t) - 1 else t[-1]
        for ax in axes:
            if section % 2 == 0:
                ax.axvspan(left, right, color=COLORS["shade"], alpha=0.58, zorder=0)
            if section > 1:
                ax.axvline(left, color="#98A2B3", linewidth=0.8, alpha=0.75, zorder=1)


def _equilibrium_references(tau):
    """Return theta references aligned with the logged target equilibrium IDs."""
    theta1_ref = np.asarray([EQUILIBRIUM_POINTS[int(idx)]["theta1"] for idx in tau])
    theta2_ref = np.asarray([EQUILIBRIUM_POINTS[int(idx)]["theta2"] for idx in tau])
    return wrap_angle(theta1_ref), wrap_angle(theta2_ref)


def _save_figure(fig, save_path):
    if save_path is None:
        return

    save_path = Path(save_path)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.04)
    print(f"[INFO] Plot saved: {save_path}")


def _edge_label(ax, y, text, color, *, fontsize=12, va="center"):
    """Add a compact label at the right edge of an axis (not a legend)."""
    if np.isfinite(y):
        ax.text(
            0.99,
            y,
            text,
            transform=ax.get_yaxis_transform(),
            ha="right",
            va=va,
            color=color,
            fontsize=fontsize,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 0.7},
            zorder=6,
        )


def plot_transitions(log, mode_name, save_path=None, include_angles=False):
    """Create a transition-test figure; arm angles are optional."""
    if len(log["t"]) == 0:
        raise ValueError("The transition log is empty; nothing can be plotted.")

    configure_plot_style()
    n_axes = 4 if include_angles else 3
    height_ratios = [1.1, 1.45, 1.05, 0.65] if include_angles else [1.15, 1.1, 0.65]
    figure_height = 9.3 if include_angles else 7.2
    fig, axes = plt.subplots(
        n_axes,
        1,
        figsize=(14.5, figure_height),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.1, top=0.97, hspace=0.12)

    ax_y = axes[0]
    ax_theta = axes[1] if include_angles else None
    ax_u = axes[2] if include_angles else axes[1]
    ax_tau = axes[3] if include_angles else axes[2]

    t = log["t"]
    y_max = CONSTRAINTS["y_max"]
    u_min, u_max = CONSTRAINTS["u_min"], CONSTRAINTS["u_max"]

    _shade_transition_sections(axes, log)

    # Cart position and safety envelope.
    ax_y.fill_between(t, -y_max, y_max, color=COLORS["cart"], alpha=0.08)
    ax_y.plot(t, log["y"], color=COLORS["cart"], zorder=3)
    ax_y.axhline(+y_max, color=COLORS["constraint"], linestyle="--", linewidth=1.2)
    ax_y.axhline(-y_max, color=COLORS["constraint"], linestyle="--", linewidth=1.2)
    ax_y.set_ylim(-1.2 * y_max, 1.2 * y_max)
    _setup_axis(ax_y, "y [m]")
    _edge_label(ax_y, +y_max, r"$+y_{\max}$", COLORS["constraint"], va="bottom")
    _edge_label(ax_y, -y_max, r"$-y_{\max}$", COLORS["constraint"], va="top")

    if include_angles:
        # Angular states and their references. Direct labels replace a legend.
        th1_ref, th2_ref = _equilibrium_references(log["tau"])
        ax_theta.plot(t, log["th1"], color=COLORS["theta1"], zorder=3)
        ax_theta.plot(t, log["th2"], color=COLORS["theta2"], zorder=3)
        ax_theta.step(
            t,
            th1_ref,
            where="post",
            color=COLORS["theta1"],
            linestyle="--",
            linewidth=1.15,
            alpha=0.75,
        )
        ax_theta.step(
            t,
            th2_ref,
            where="post",
            color=COLORS["theta2"],
            linestyle="--",
            linewidth=1.15,
            alpha=0.75,
        )
        ax_theta.set_ylim(-np.pi - 0.18, np.pi + 0.18)
        ax_theta.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax_theta.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        _setup_axis(ax_theta, "Arm angles [rad]")
        ax_theta.text(
            0.01,
            0.94,
            "Solid: state    Dashed: reference",
            transform=ax_theta.transAxes,
            ha="left",
            va="top",
            color="#475467",
            fontsize=12,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.9},
        )
        _edge_label(ax_theta, log["th1"][-1], r"$\theta_1$", COLORS["theta1"])
        _edge_label(ax_theta, log["th2"][-1], r"$\theta_2$", COLORS["theta2"])

    # Applied acceleration and input constraints.
    ax_u.fill_between(t, u_min, u_max, color=COLORS["input"], alpha=0.07)
    ax_u.plot(t, log["u"], color=COLORS["input"], zorder=3)
    ax_u.axhline(u_min, color=COLORS["constraint"], linestyle="--", linewidth=1.1)
    ax_u.axhline(u_max, color=COLORS["constraint"], linestyle="--", linewidth=1.1)
    ax_u.set_ylim(1.12 * u_min, 1.12 * u_max)
    _setup_axis(ax_u, r"u [m/s$^2$]")
    _edge_label(ax_u, u_max, r"$u_{\max}$", COLORS["constraint"], va="bottom")
    _edge_label(ax_u, u_min, r"$u_{\min}$", COLORS["constraint"], va="top")

    # Target-equilibrium command.
    ax_tau.step(t, log["tau"], where="post", color=COLORS["target"], linewidth=2.1)
    ax_tau.set_yticks([0, 1, 2, 3])
    ax_tau.set_yticklabels(["EP0", "EP1", "EP2", "EP3"])
    ax_tau.set_ylim(-0.35, 3.55)
    ax_tau.yaxis.set_major_locator(MaxNLocator(integer=True))
    _setup_axis(ax_tau, "Target EP")
    ax_tau.set_xlabel("Time [s]")

    total_time = t[-1] if len(t) else 0.0
    # fig.suptitle(
    #     f"Transition-control evaluation — {_mode_label(mode_name)}",
    #     x=0.05,
    #     ha="left",
    #     fontsize=14,
    #     fontweight="bold",
    #     y=0.975,
    # )
    angle_note = "  |  arm-angle subplot included" if include_angles else ""
    # fig.text(
    #     0.05,
    #     0.940,
    #     (
    #         f"Sequence of {len(TRANSITION_SEQUENCE)} commanded equilibrium transitions"
    #         f"  |  total duration: {total_time:.1f} s{angle_note}"
    #     ),
    #     ha="left",
    #     va="top",
    #     color="#475467",
    #     fontsize=10,
    # )

    _save_figure(fig, save_path)
    return fig


def plot_recovery(log, mode_name, save_path=None, include_angles=False):
    """Create a recovery-test figure; arm angles are optional."""
    if len(log["t"]) == 0:
        raise ValueError("The recovery log is empty; nothing can be plotted.")

    configure_plot_style()
    n_axes = 3 if include_angles else 2
    height_ratios = [1.1, 1.4, 1.05] if include_angles else [1.1, 1.05]
    figure_height = 7.8 if include_angles else 5.8
    fig, axes = plt.subplots(
        n_axes,
        1,
        figsize=(13.5, figure_height),
        sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.09, right=0.985, bottom=0.10, top=0.875, hspace=0.095)

    ax_y = axes[0]
    ax_theta = axes[1] if include_angles else None
    ax_u = axes[2] if include_angles else axes[1]

    t = log["t"]
    y_max = CONSTRAINTS["y_max"]
    u_min, u_max = CONSTRAINTS["u_min"], CONSTRAINTS["u_max"]

    disturbed_idx = np.flatnonzero(log["disturbed"])
    disturbance_time = t[disturbed_idx[0]] if disturbed_idx.size else None

    # Cart position and safety envelope.
    ax_y.fill_between(t, -y_max, y_max, color=COLORS["cart"], alpha=0.08)
    ax_y.plot(t, log["y"], color=COLORS["cart"], zorder=3)
    ax_y.axhline(+y_max, color=COLORS["constraint"], linestyle="--", linewidth=1.2)
    ax_y.axhline(-y_max, color=COLORS["constraint"], linestyle="--", linewidth=1.2)
    ax_y.set_ylim(-1.2 * y_max, 1.2 * y_max)
    _setup_axis(ax_y, "Cart position\n $y$ [m]")
    _edge_label(ax_y, +y_max, r"$+y_{\max}$", COLORS["constraint"], va="bottom")
    _edge_label(ax_y, -y_max, r"$-y_{\max}$", COLORS["constraint"], va="top")

    if include_angles:
        ax_theta.plot(t, log["th1"], color=COLORS["theta1"], zorder=3)
        ax_theta.plot(t, log["th2"], color=COLORS["theta2"], zorder=3)
        ax_theta.set_ylim(-np.pi - 0.18, np.pi + 0.18)
        ax_theta.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
        ax_theta.set_yticklabels([r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
        _setup_axis(ax_theta, "Arm angles\n [rad]")
        _edge_label(ax_theta, log["th1"][-1], r"$\theta_1$", COLORS["theta1"])
        _edge_label(ax_theta, log["th2"][-1], r"$\theta_2$", COLORS["theta2"])

    # Applied acceleration and input limits.
    ax_u.fill_between(t, u_min, u_max, color=COLORS["input"], alpha=0.07)
    ax_u.plot(t, log["u"], color=COLORS["input"], zorder=3)
    ax_u.axhline(u_min, color=COLORS["constraint"], linestyle="--", linewidth=1.1)
    ax_u.axhline(u_max, color=COLORS["constraint"], linestyle="--", linewidth=1.1)
    ax_u.set_ylim(1.12 * u_min, 1.12 * u_max)
    _setup_axis(ax_u, "Acceleration\n $u$ [m/s$^2$]")
    _edge_label(ax_u, u_max, r"$u_{\max}$", COLORS["constraint"], va="bottom")
    _edge_label(ax_u, u_min, r"$u_{\min}$", COLORS["constraint"], va="top")
    ax_u.set_xlabel("Time [s]")

    if disturbance_time is not None:
        band = max(0.02 * (t[-1] - t[0]), 2 * SIM_PARAMS["control_dt"])
        for ax in axes:
            ax.axvspan(
                disturbance_time - band / 2,
                disturbance_time + band / 2,
                color=COLORS["disturbance"],
                alpha=0.10,
                zorder=1,
            )
            ax.axvline(
                disturbance_time,
                color=COLORS["disturbance"],
                linestyle="--",
                linewidth=1.4,
                zorder=4,
            )
        ax_y.annotate(
            "Impulse disturbance",
            xy=(disturbance_time, 0.90 * y_max),
            xytext=(10, 0),
            textcoords="offset points",
            color=COLORS["disturbance"],
            fontsize=11,
            va="center",
            fontweight="semibold",
        )

    total_time = t[-1] if len(t) else 0.0
    # fig.suptitle(
    #     f"Disturbance-recovery evaluation — {_mode_label(mode_name)}",
    #     x=0.05,
    #     ha="left",
    #     fontsize=14,
    #     fontweight="bold",
    #     y=0.975,
    # )
    # subtitle = f"Total duration: {total_time:.1f} s"
    # if disturbance_time is not None:
    #     subtitle += f"  |  velocity impulse applied at t = {disturbance_time:.2f} s"
    # if include_angles:
    #     subtitle += "  |  arm-angle subplot included"
    # fig.text(0.05, 0.940, subtitle, ha="left", va="top", color="#475467", fontsize=10)

    _save_figure(fig, save_path)
    return fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--mode",
        choices=["vanilla", "reward_shaper", "safety_filter", "hierarchical"],
        default="vanilla",
    )
    parser.add_argument(
        "--disturbance",
        action="store_true",
        help="Run the recovery test instead of the transition test.",
    )
    parser.add_argument(
        "--plot-angles",
        action="store_true",
        help="Include the arm-angle subplot. By default only position, input, and command are plotted.",
    )
    parser.add_argument("--out", type=str, default="./eval_output")
    parser.add_argument(
        "--format",
        choices=["svg", "pdf", "png"],
        default="pdf",
        help="Output format. SVG and PDF are best suited to papers and theses.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the figure without opening an interactive Matplotlib window.",
    )
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True, parents=True)

    print(f"[INFO] Loading model: {args.model}")
    model = TQC.load(args.model)

    mpc = NMPCController() if args.mode != "vanilla" else None

    angle_suffix = "_angles" if args.plot_angles else ""
    if args.disturbance:
        log = evaluate_recovery(model, args.mode, mpc)
        plot_recovery(
            log,
            args.mode,
            out / f"recovery_{args.mode}{angle_suffix}.{args.format}",
            include_angles=args.plot_angles,
        )
    else:
        log = evaluate_transitions(model, args.mode, mpc)
        plot_transitions(
            log,
            args.mode,
            out / f"transitions_{args.mode}{angle_suffix}.{args.format}",
            include_angles=args.plot_angles,
        )

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
