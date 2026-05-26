"""Training entry point.

Usage examples:
    # Start fresh with 8 parallel envs (default, RTX 4060):
    python -m rl_agent.train

    # Scale up to 64 envs for A6000:
    python -m rl_agent.train --num_envs 64

    # Resume from checkpoint:
    python -m rl_agent.train --resume rl_agent/checkpoints/checkpoint_step1000000.pt

    # Short smoke run:
    python -m rl_agent.train --total_steps 10000 --num_envs 1
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

# Ensure repo root is on path.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("FLOW_HEADLESS", "1")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train hierarchical RL agent for Flow Game")
    p.add_argument("--num_envs",     type=int,   default=None,
                   help="Parallel envs (default: config.NUM_ENVS)")
    p.add_argument("--total_steps",  type=int,   default=None,
                   help="Total env steps (default: config.TOTAL_TIMESTEPS)")
    p.add_argument("--checkpoint_dir", type=str, default="rl_agent/checkpoints")
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--seed",         type=int,   default=0)
    p.add_argument("--watch",        action="store_true",
                   help="Open a live game window showing the agent play while training")
    p.add_argument("--watch_windowed", action="store_true",
                   help="Render the watch window in windowed mode (default: fullscreen)")
    return p.parse_args()


def _launch_watcher(checkpoint_dir: str, windowed: bool) -> "subprocess.Popen":
    """Spawn the watch.py renderer in its own process."""
    import subprocess
    cmd = [
        sys.executable, "-m", "rl_agent.watch",
        "--checkpoint_dir", checkpoint_dir,
        "--reload_every", "15",
    ]
    if windowed:
        cmd.append("--windowed")
    proc = subprocess.Popen(cmd, cwd=_REPO_ROOT)
    return proc


def main() -> None:
    args = parse_args()

    from rl_agent import config as cfg
    n_envs      = args.num_envs    or cfg.NUM_ENVS
    total_steps = args.total_steps or cfg.TOTAL_TIMESTEPS

    print(f"Device  : {cfg.DEVICE}")
    print(f"Envs    : {n_envs}")
    print(f"Steps   : {total_steps:,}")
    print(f"Seed    : {args.seed}")
    if args.resume:
        print(f"Resume  : {args.resume}")
    if args.watch:
        print(f"Watch   : enabled (live renderer window)")
    print()

    # Launch watcher before training so it opens while the first checkpoint
    # is being written. It will wait and reload automatically.
    watcher_proc  = None
    warmup_ckpt   = args.resume   # carry forward any existing resume path

    if args.watch:
        import glob, time as _time

        # Check if a checkpoint already exists; if not, do a tiny warm-up so
        # the watcher has something to load immediately.
        existing = glob.glob(os.path.join(args.checkpoint_dir, "checkpoint_*.pt"))
        if not existing:
            print("[watch] doing 2048-step warm-up to create initial checkpoint...")
            from rl_agent.hierarchical import HierarchicalTrainer
            warm = HierarchicalTrainer(
                n_envs=1, total_timesteps=2048,
                checkpoint_dir=args.checkpoint_dir,
                resume=args.resume, seed=args.seed,
            )
            warm.train()
            warm.envs.close()
            del warm
            # Use the warm-up checkpoint as the starting point for main training.
            warmup_ckpt = os.path.join(args.checkpoint_dir, "checkpoint_final.pt")
        else:
            print("[watch] found existing checkpoint — skipping warm-up")

        watcher_proc = _launch_watcher(args.checkpoint_dir, args.watch_windowed)
        print(f"[watch] watcher PID={watcher_proc.pid}  "
              f"(close the window or Ctrl+C to stop)")
        _time.sleep(2)   # give watcher window time to appear

    try:
        from rl_agent.hierarchical import HierarchicalTrainer
        trainer = HierarchicalTrainer(
            n_envs           = n_envs,
            total_timesteps  = total_steps,
            checkpoint_dir   = args.checkpoint_dir,
            resume           = warmup_ckpt,
            seed             = args.seed,
        )
        trainer.train()
    finally:
        if watcher_proc is not None and watcher_proc.poll() is None:
            print("[watch] training done — watcher still open, press ESC to close it.")


if __name__ == "__main__":
    main()
