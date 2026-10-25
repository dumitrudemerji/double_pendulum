"""
Script di training dell'agente RL nelle 4 modalità.

Usa TQC (Truncated Quantile Critics) da sb3-contrib, lo stesso algoritmo
di Lee et al. 2025.

Esempi:
  # CPU (default, 1 environment)
  python -m training.train --mode vanilla

  # GPU automatica
  python -m training.train --mode hierarchical --device auto

  # 4 ambienti in parallelo (SubprocVecEnv)
  python -m training.train --mode vanilla --n-envs 4

  # 8 ambienti + GPU
  python -m training.train --mode reward_shaper --n-envs 8 --device cuda

  # Apple Silicon
  python -m training.train --mode vanilla --device mps

NOTA SU --n-envs:
  - n-envs=1 usa DummyVecEnv (sequenziale, basso overhead)
  - n-envs>1 usa SubprocVecEnv (parallelo, un processo per env)
  - Ogni worker crea la propria istanza NMPC se mode != 'vanilla'
    -> maggiore consumo di RAM, ma vero parallelismo
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import TQC
from stable_baselines3.common.callbacks import (
    BaseCallback, CallbackList, EvalCallback, CheckpointCallback
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from controllers.environment import DoublePendulumEnv
from controllers.nmpc import NMPCController
from configs.config import RL_PARAMS


# ──────────────────────────────────────────────────────────────
# Callback: salva reward episodio + mean rolling ogni N step
# ──────────────────────────────────────────────────────────────

class RewardLoggerCallback(BaseCallback):
    """
    Logga la reward ad ogni episodio su un CSV dedicato.

    Colonne salvate:
      episode    : numero episodio
      timestep   : step globale di training
      reward     : reward totale dell'episodio
      ep_length  : lunghezza episodio (step)
      elapsed_s  : secondi dall'inizio del training
    """

    def __init__(self, log_path, verbose=0):
        super().__init__(verbose)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._episode = 0
        self._ep_reward = 0.0
        self._ep_length = 0
        self._start_time = None
        self._csv_file = None
        self._writer = None

    def _on_training_start(self):
        self._start_time = time.time()
        self._ep_reward = 0.0
        self._ep_length = 0
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = open(self.log_path, 'w', newline='')
            self._writer = csv.writer(self._csv_file)
            self._writer.writerow(
                ['episode', 'timestep', 'reward', 'ep_length', 'elapsed_s']
            )
            self._csv_file.flush()
            print(f"[INFO] RewardLogger attivo -> {self.log_path}")
        except Exception as e:
            print(f"[WARN] RewardLogger: impossibile aprire {self.log_path}: {e}")

    def _on_step(self):
        # Accumula reward e length step-by-step
        rewards = self.locals.get('rewards', [0.0])
        dones   = self.locals.get('dones',   [False])

        for reward, done in zip(rewards, dones):
            self._ep_reward += float(reward)
            self._ep_length += 1

            if done and self._writer is not None:
                self._episode += 1
                elapsed = time.time() - self._start_time
                self._writer.writerow([
                    self._episode,
                    self.num_timesteps,
                    round(self._ep_reward, 6),
                    self._ep_length,
                    round(elapsed, 2),
                ])
                self._csv_file.flush()
                # Reset per nuovo episodio
                self._ep_reward = 0.0
                self._ep_length = 0

        return True

    def _on_training_end(self):
        if self._csv_file:
            self._csv_file.close()
            print(f"[INFO] RewardLogger: {self._episode} episodi salvati in {self.log_path}")


# ──────────────────────────────────────────────────────────────
# Device resolution
# ──────────────────────────────────────────────────────────────

def resolve_device(device_arg):
    """
    Risolve la stringa di dispositivo richiesta dall'utente.

    Args:
        device_arg: 'auto' | 'cpu' | 'cuda' | 'cuda:0' | 'mps' | None

    Returns:
        device: stringa torch valida e disponibile
        info_msg: messaggio descrittivo per il log
    """
    if device_arg in (None, 'auto'):
        if torch.cuda.is_available():
            device = 'cuda'
            n_gpus = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            info = f"CUDA disponibile: {n_gpus} GPU rilevate, uso '{gpu_name}'"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
            info = "Apple MPS disponibile, uso GPU integrata"
        else:
            device = 'cpu'
            info = "Nessuna GPU rilevata, uso CPU"
        return device, info

    if device_arg.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"Richiesto '{device_arg}' ma CUDA non è disponibile. "
                "Installa torch con supporto CUDA o usa --device cpu."
            )
        if ':' in device_arg:
            idx = int(device_arg.split(':')[1])
            if idx >= torch.cuda.device_count():
                raise RuntimeError(
                    f"GPU {idx} non esiste. "
                    f"Disponibili: 0..{torch.cuda.device_count() - 1}"
                )
        gpu_name = torch.cuda.get_device_name(
            int(device_arg.split(':')[1]) if ':' in device_arg else 0
        )
        return device_arg, f"Uso GPU CUDA: {gpu_name}"

    if device_arg == 'mps':
        if not (hasattr(torch.backends, 'mps') and
                torch.backends.mps.is_available()):
            raise RuntimeError(
                "Richiesto 'mps' ma non disponibile (richiede macOS + Apple Silicon)"
            )
        return 'mps', "Uso Apple MPS (GPU integrata)"

    if device_arg == 'cpu':
        return 'cpu', "Uso CPU (forzato)"

    raise ValueError(f"device_arg sconosciuto: {device_arg}")


# ──────────────────────────────────────────────────────────────
# Environment factory
# ──────────────────────────────────────────────────────────────

def make_env_fn(mode, seed=0, worker_idx=0, needs_mpc_in_worker=False):
    """
    Factory di environment per VecEnv.

    Args:
        mode:    modalità (vanilla/reward_shaper/safety_filter/hierarchical)
        seed:    seed base (verrà combinato con worker_idx)
        worker_idx: indice del worker (per seed differenziato)
        needs_mpc_in_worker: se True, ogni worker crea il proprio NMPC.
                              True per SubprocVecEnv con mode != 'vanilla'.
                              False per DummyVecEnv (NMPC condiviso passato fuori).
    """
    def _init():
        # Crea NMPC dentro il worker se richiesto.
        # Necessario per SubprocVecEnv perché l'oggetto NMPC (CasADi + IPOPT)
        # non è pickleable e non può essere trasferito tra processi.
        mpc_local = None
        if needs_mpc_in_worker and mode != 'vanilla':
            from controllers.nmpc import NMPCController
            mpc_local = NMPCController(verbose=False)

        env = DoublePendulumEnv(
            mode=mode,
            mpc_controller=mpc_local,
            randomize_tau=True,
        )
        env.reset(seed=seed + worker_idx * 1000)
        return env
    return _init


def build_vec_env(mode, n_envs, log_dir_base=None, seed=0,
                  is_eval=False):
    """
    Costruisce il VecEnv appropriato in base a n_envs.

    n_envs == 1 -> DummyVecEnv (sequenziale, NMPC condiviso)
    n_envs >  1 -> SubprocVecEnv (1 processo per env, NMPC indipendenti)

    Args:
        mode:         modalità di training
        n_envs:       numero di ambienti paralleli
        log_dir_base: directory base per Monitor (solo se n_envs == 1
                      o per il primo worker)
        seed:         seed di base
        is_eval:      se True, riduce verbose nei messaggi

    Returns:
        (vec_env, shared_mpc)
        shared_mpc è None quando si usa SubprocVecEnv
    """
    label = "eval" if is_eval else "train"

    if n_envs == 1:
        # Caso sequenziale: NMPC condiviso
        mpc_shared = None
        if mode != 'vanilla':
            if not is_eval:
                print(f"[INFO] Costruzione NMPC condiviso per mode='{mode}'...")
            mpc_shared = NMPCController(verbose=False)

        def _make():
            env = DoublePendulumEnv(
                mode=mode, mpc_controller=mpc_shared,
                randomize_tau=True,
            )
            env.reset(seed=seed)
            if log_dir_base is not None:
                env = Monitor(env, str(log_dir_base))
            return env

        return DummyVecEnv([_make]), mpc_shared

    # Caso parallelo: SubprocVecEnv
    if not is_eval:
        print(f"[INFO] Costruzione SubprocVecEnv ({label}) "
              f"con {n_envs} workers...")
        if mode != 'vanilla':
            print(f"[INFO] Ogni worker creerà il proprio NMPC "
                  f"-> RAM ~{n_envs}× rispetto al setup sequenziale")

    env_fns = [
        make_env_fn(
            mode=mode,
            seed=seed,
            worker_idx=i,
            needs_mpc_in_worker=True,
        )
        for i in range(n_envs)
    ]

    # start_method='spawn' è più portabile (funziona anche su macOS e Windows)
    # 'fork' è più veloce ma può causare problemi con CUDA e CasADi
    vec_env = SubprocVecEnv(env_fns, start_method='spawn')
    return vec_env, None


# ──────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────

def train(mode='vanilla', total_timesteps=None, run_name=None,
          device='auto', n_envs=1):
    """
    Train del modello TQC con la modalità e parallelismo specificati.

    Args:
        mode:            'vanilla' | 'reward_shaper' | 'safety_filter' | 'hierarchical'
        total_timesteps: numero totale di step (default dal config)
        run_name:        nome della cartella di log (default: 'tqc_<mode>')
        device:          'auto' | 'cpu' | 'cuda' | 'cuda:N' | 'mps'
        n_envs:          numero di environment paralleli
                         (1 = DummyVecEnv, >1 = SubprocVecEnv)
    """
    if total_timesteps is None:
        total_timesteps = RL_PARAMS['total_timesteps']
    if run_name is None:
        run_name = f"tqc_{mode}"

    # Device
    device, info = resolve_device(device)
    print(f"[INFO] {info}")
    print(f"[INFO] Dispositivo selezionato: {device}")
    print(f"[INFO] Numero di environment paralleli: {n_envs} "
          f"({'SubprocVecEnv' if n_envs > 1 else 'DummyVecEnv'})")

    log_dir = Path(f"./logs/{run_name}")
    model_dir = Path(f"./models_saved/{run_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Train env: parallelo se n_envs > 1
    env, _ = build_vec_env(
        mode=mode,
        n_envs=n_envs,
        log_dir_base=log_dir / "train" if n_envs == 1 else None,
        seed=0,
        is_eval=False,
    )

    # Eval env: sempre sequenziale (basta 1 env per la valutazione periodica)
    eval_env, _ = build_vec_env(
        mode=mode,
        n_envs=1,
        log_dir_base=log_dir / "eval",
        seed=42,
        is_eval=True,
    )

    policy_kwargs = dict(
        net_arch=dict(
            pi=RL_PARAMS['policy_layers'],
            qf=RL_PARAMS['critic_layers'],
        ),
        n_critics=RL_PARAMS['n_critics'],
        n_quantiles=RL_PARAMS['n_quantiles'],
    )

    # tensorboard opzionale
    try:
        import tensorboard  # noqa: F401
        tb_log = str(log_dir / "tb")
    except ImportError:
        print("[INFO] tensorboard non installato, log su file standard")
        tb_log = None

    model = TQC(
        policy='MlpPolicy',
        env=env,
        learning_rate=RL_PARAMS['learning_rate'],
        buffer_size=RL_PARAMS['buffer_size'],
        batch_size=RL_PARAMS['batch_size'],
        gamma=RL_PARAMS['gamma'],
        tau=RL_PARAMS['tau'],
        policy_kwargs=policy_kwargs,
        device=device,
        verbose=1,
        tensorboard_log=tb_log,
    )

    actual_device = next(model.policy.parameters()).device
    print(f"[INFO] Modello istanziato su: {actual_device}")

    # Callbacks
    reward_csv = log_dir / "reward_log.csv"
    reward_callback = RewardLoggerCallback(log_path=reward_csv)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir / "eval_log"),
        eval_freq=1_000,
        n_eval_episodes=5,
        deterministic=True,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=1_000,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="tqc",
    )

    print(f"[INFO] Training: mode={mode}, steps={total_timesteps}, "
          f"device={actual_device}")
    print(f"[INFO] Reward log salvato in: {reward_csv}")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[
            reward_callback,
            eval_callback,
            checkpoint_callback,
        ],
        progress_bar=True,
    )

    model.save(str(model_dir / "final"))
    print(f"[INFO] Modello salvato in: {model_dir / 'final'}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode',
        choices=['vanilla', 'reward_shaper', 'safety_filter', 'hierarchical'],
        default='vanilla',
        help='Modalità di integrazione MPC+RL',
    )
    parser.add_argument('--steps', type=int, default=None,
                        help='Numero di step di training (default da config)')
    parser.add_argument('--name', type=str, default=None,
                        help='Nome del run (default: tqc_<mode>)')
    parser.add_argument(
        '--device', type=str, default='auto',
        help=(
            "Dispositivo PyTorch: 'auto' (default), 'cpu', "
            "'cuda', 'cuda:0', 'cuda:1', ... oppure 'mps' (Apple Silicon)"
        ),
    )
    parser.add_argument(
        '--n-envs', type=int, default=1,
        help=(
            "Numero di ambienti paralleli. "
            "1 = DummyVecEnv (sequenziale). "
            ">1 = SubprocVecEnv (1 processo per env). "
            "Suggerito: numero di CPU core - 1. "
            "Attenzione: con mode != 'vanilla', ogni worker crea "
            "il proprio NMPC -> uso RAM proporzionale a n_envs."
        ),
    )
    args = parser.parse_args()
    train(
        mode=args.mode,
        total_timesteps=args.steps,
        run_name=args.name,
        device=args.device,
        n_envs=args.n_envs,
    )