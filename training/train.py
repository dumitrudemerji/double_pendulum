"""
Training script of the RL agent in the 4 modes.

Uses TQC (Truncated Quantile Critics) from sb3-contrib, the same algorithm
as Lee et al. 2025.

Examples:
  # Training from zero
  python -m training.train --mode vanilla

  # GPU + parallel
  python -m training.train --mode hierarchical --device auto --n-envs 4

  # Resume automatically from the most recent checkpoint
  python -m training.train --mode vanilla --resume auto

  # Resume from a specific checkpoint
  python -m training.train --mode vanilla \\
      --resume models_saved/tqc_vanilla/checkpoints/tqc_500000_steps.zip

  # Resume from the final model and continue for another 1M steps
  python3 -m training.train --mode vanilla --steps 1000000 \\
      --resume models_saved/tqc_safety_filter/final.zip

NOTA SU --resume:
  - 'auto'          → search the most recent checkpoint in models_saved/<run>/
  - 'best'          → load the best model (best_model.zip)
  - 'final'         → load final.zip
  - <path>.zip      → load the specific model
  - The replay buffer is loaded if available (adjacent .pkl file)
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
# Callback: save episode reward + mean rolling every N step
# ──────────────────────────────────────────────────────────────

class RewardLoggerCallback(BaseCallback):
    """
    Log the reward at each episode in a dedicated CSV file.

    Columns saved:
      episode    : episode number
      timestep   : training timestep (global)
      reward     : total reward of the episode
      ep_length  : episode length (steps)
      elapsed_s  : seconds from the start of training
    """

    def __init__(self, log_path, append=False, verbose=0):
        super().__init__(verbose)
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._episode = 0
        self._ep_reward = 0.0
        self._ep_length = 0
        self._start_time = None
        self._csv_file = None
        self._writer = None
        self._append = append

    def _on_training_start(self):
        self._start_time = time.time()
        # In append mode (resume), conserva gli episodi già loggati
        if self._append and self.log_path.exists():
            try:
                with open(self.log_path, 'r') as f:
                    existing = [r for r in csv.DictReader(f)]
                self._episode = len(existing)
            except Exception:
                self._episode = 0
        else:
            self._episode = 0
        self._ep_reward = 0.0
        self._ep_length = 0
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            mode = 'a' if (self._append and self.log_path.exists()) else 'w'
            self._csv_file = open(self.log_path, mode, newline='')
            self._writer = csv.writer(self._csv_file)
            if mode == 'w':
                self._writer.writerow(
                    ['episode', 'timestep', 'reward', 'ep_length', 'elapsed_s']
                )
            else:
                print(f"[INFO] RewardLogger in modalità APPEND "
                      f"(esistenti: {self._episode} episodi)")
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
# CheckpointCallback custom: saves also the replay buffer
# ──────────────────────────────────────────────────────────────

class CheckpointWithBufferCallback(BaseCallback):
    """
    Salva periodicamente:
      - il modello completo (.zip)
      - il replay buffer (.pkl)

    Il replay buffer è fondamentale per riprendere il training
    di algoritmi off-policy come TQC senza perdere l'esperienza
    accumulata.
    """

    def __init__(self, save_freq, save_path, name_prefix='tqc',
                 save_replay_buffer=True, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = Path(save_path)
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.save_freq == 0:
            steps = self.num_timesteps
            model_path = self.save_path / f"{self.name_prefix}_{steps}_steps.zip"
            self.model.save(model_path)

            if self.save_replay_buffer:
                buffer_path = self.save_path / f"{self.name_prefix}_{steps}_steps_buffer.pkl"
                try:
                    self.model.save_replay_buffer(buffer_path)
                except Exception as e:
                    print(f"[WARN] Impossibile salvare buffer: {e}")

            if self.verbose:
                print(f"[CHK] Checkpoint salvato @ step {steps}: {model_path.name}")
        return True


# ──────────────────────────────────────────────────────────────
# Checkpoint resolution
# ──────────────────────────────────────────────────────────────

def resolve_checkpoint(resume_arg, run_name):
    """
    Risolve il path del checkpoint da cui riprendere.

    Args:
        resume_arg: None, 'auto', 'best', 'final', oppure un path
        run_name:   nome del run (per cercare in models_saved/<run>/)

    Returns:
        (model_path, buffer_path, info_msg)
        model_path è None se non c'è nulla da caricare
        buffer_path è None se il buffer non esiste
    """
    if resume_arg is None:
        return None, None, "Nessun resume (training da zero)"

    model_dir = Path(f"./models_saved/{run_name}")

    # Caso 1: 'auto' → cerca il checkpoint più recente
    if resume_arg == 'auto':
        if not model_dir.exists():
            return None, None, ("Auto-resume: nessuna cartella esistente "
                                 f"per '{run_name}', training da zero")

        # Cerca prima i checkpoint numerati
        chk_dir = model_dir / 'checkpoints'
        if chk_dir.exists():
            checkpoints = sorted(
                chk_dir.glob(f'*_steps.zip'),
                key=lambda p: int(p.stem.split('_')[-2])
            )
            if checkpoints:
                latest = checkpoints[-1]
                buf = latest.parent / (latest.stem + '_buffer.pkl')
                buf_path = buf if buf.exists() else None
                steps = int(latest.stem.split('_')[-2])
                return (str(latest), str(buf_path) if buf_path else None,
                        f"Auto-resume: checkpoint più recente {latest.name} (step {steps})")

        # Fallback: final.zip
        final = model_dir / 'final.zip'
        if final.exists():
            return str(final), None, f"Auto-resume: trovato {final.name}"

        return None, None, f"Auto-resume: nessun checkpoint in {model_dir}, training da zero"

    # Caso 2: 'best' → carica best_model.zip
    if resume_arg == 'best':
        best = model_dir / 'best' / 'best_model.zip'
        if not best.exists():
            raise FileNotFoundError(f"Best model non trovato: {best}")
        return str(best), None, f"Resume da best model: {best}"

    # Caso 3: 'final' → carica final.zip
    if resume_arg == 'final':
        final = model_dir / 'final.zip'
        if not final.exists():
            raise FileNotFoundError(f"Final model non trovato: {final}")
        return str(final), None, f"Resume da final: {final}"

    # Caso 4: path esplicito
    path = Path(resume_arg)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {path}")

    # Cerca buffer affiancato
    buf = path.parent / (path.stem + '_buffer.pkl')
    buf_path = str(buf) if buf.exists() else None

    return str(path), buf_path, f"Resume da {path}"


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
          device='auto', n_envs=1, resume=None,
          checkpoint_freq=100_000, save_buffer=True):
    """
    Train del modello TQC con la modalità e parallelismo specificati.

    Args:
        mode:            'vanilla' | 'reward_shaper' | 'safety_filter' | 'hierarchical'
        total_timesteps: numero totale di step (default dal config)
        run_name:        nome della cartella di log (default: 'tqc_<mode>')
        device:          'auto' | 'cpu' | 'cuda' | 'cuda:N' | 'mps'
        n_envs:          numero di environment paralleli
        resume:          None | 'auto' | 'best' | 'final' | path.zip
                         Se non None, carica il modello e riprende il training.
        checkpoint_freq: ogni quanti step salvare un checkpoint
        save_buffer:     se True, salva anche il replay buffer ad ogni checkpoint
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

    # Risolvi checkpoint da cui riprendere
    model_path, buffer_path, resume_msg = resolve_checkpoint(resume, run_name)
    print(f"[INFO] {resume_msg}")
    is_resume = model_path is not None

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

    # Eval env: sempre sequenziale
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

    # ── Costruzione/caricamento modello ──
    if is_resume:
        # RESUME: carica il modello esistente
        print(f"[INFO] Caricamento modello da: {model_path}")
        model = TQC.load(
            model_path,
            env=env,
            device=device,
            tensorboard_log=tb_log,
            # I parametri come learning_rate vengono ricaricati dal modello;
            # se vuoi sovrascriverli, passali con custom_objects
        )
        print(f"[INFO] Modello caricato: {model.num_timesteps} step "
              f"già completati")

        # Carica replay buffer se disponibile
        if buffer_path:
            print(f"[INFO] Caricamento replay buffer da: {buffer_path}")
            try:
                model.load_replay_buffer(buffer_path)
                print(f"[INFO] Buffer caricato: "
                      f"{model.replay_buffer.size()} transizioni")
            except Exception as e:
                print(f"[WARN] Buffer non caricato ({e}), training continua "
                      f"con buffer vuoto")
        else:
            print("[WARN] Nessun replay buffer trovato. "
                  "Il training riprende con buffer vuoto — "
                  "le prime fasi saranno simili a un cold-start.")
    else:
        # TRAINING DA ZERO
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
    print(f"[INFO] Modello pronto su: {actual_device}")

    # Callbacks (in append mode se resume)
    reward_csv = log_dir / "reward_log.csv"
    reward_callback = RewardLoggerCallback(
        log_path=reward_csv,
        append=is_resume,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(model_dir / "best"),
        log_path=str(log_dir / "eval_log"),
        eval_freq=10000,
        n_eval_episodes=5,
        deterministic=True,
    )

    # CheckpointCallback custom con replay buffer
    checkpoint_callback = CheckpointWithBufferCallback(
        save_freq=checkpoint_freq,
        save_path=str(model_dir / "checkpoints"),
        name_prefix="tqc",
        save_replay_buffer=save_buffer,
    )

    action_word = "Resume" if is_resume else "Training"
    print(f"[INFO] {action_word}: mode={mode}, steps={total_timesteps}, "
          f"device={actual_device}")
    print(f"[INFO] Reward log: {reward_csv}")
    print(f"[INFO] Checkpoints ogni {checkpoint_freq} step "
          f"(replay buffer: {'sì' if save_buffer else 'no'})")

    model.learn(
        total_timesteps=total_timesteps,
        callback=[
            reward_callback,
            eval_callback,
            checkpoint_callback,
        ],
        progress_bar=True,
        # In resume: continua il conteggio degli step da dove eravamo
        # (non resettare a 0). In training fresh: parti da 0.
        reset_num_timesteps=not is_resume,
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
        help='Integration mode MPC+RL',
    )
    parser.add_argument('--steps', type=int, default=None,
                        help='Number of training steps (default from config)')
    parser.add_argument('--name', type=str, default=None,
                        help='Run name (default: tqc_<mode>)')
    parser.add_argument(
        '--device', type=str, default='auto',
        help=(
            "Device PyTorch: 'auto' (default), 'cpu', "
            "'cuda', 'cuda:0', 'cuda:1', ... oppure 'mps' (Apple Silicon)"
        ),
    )
    parser.add_argument(
        '--n-envs', type=int, default=1,
        help=(
            "Number of parallel environments. "
            "1 = DummyVecEnv (sequential). "
            ">1 = SubprocVecEnv (1 process per env). "
            "Suggested: number of CPU cores - 1. "
            "Attention: with mode != 'vanilla', every worker creates "
            "the proper NMPC -> use RAM proporzionale a n_envs."
        ),
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help=(
            "Resume an existing training. Values:\n"
            "  'auto'  -> last available checkpoint for <run_name>\n"
            "  'best'  -> best_model.zip\n"
            "  'final' -> final.zip\n"
            "  <path>  -> explicit path to a .zip file\n"
            "If omitted, start from zero."
        ),
    )
    parser.add_argument(
        '--checkpoint-freq', type=int, default=100000,
        help='Ogni quanti step salvare un checkpoint (default 10000)',
    )
    parser.add_argument(
        '--no-buffer', action='store_true',
        help='Non salvare il replay buffer nei checkpoint (riduce dimensione)',
    )
    args = parser.parse_args()
    train(
        mode=args.mode,
        total_timesteps=args.steps,
        run_name=args.name,
        device=args.device,
        n_envs=args.n_envs,
        resume=args.resume,
        checkpoint_freq=args.checkpoint_freq,
        save_buffer=not args.no_buffer,
    )

