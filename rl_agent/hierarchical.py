"""Hierarchical training coordinator.

Architecture:
    - Meta-policy  : selects target archetype every OPTION_PERIOD_K low-level
                     steps.  Trained with PPO using sum of low-level rewards
                     over the option period as meta-reward.
    - Low-level    : executes game actions (move, dash, shop) step-by-step.
                     Trained with standard PPO every LL_ROLLOUT_STEPS.

Parallelism:
    N parallel environments are run in separate Python processes via
    multiprocessing.  Each worker owns one FlowGameEnv.  The coordinator
    collects rollouts from all workers, flattens them, and runs one GPU
    update per policy.
"""
from __future__ import annotations

import os
import time
from multiprocessing import Process
from typing import Dict, List, Optional, Tuple

import multiprocessing as mp

import numpy as np
import torch

from rl_agent.config import (
    NUM_ENVS, OPTION_PERIOD_K, DEVICE,
    OBS_DIM, META_OBS_DIM,
    N_ARCHETYPES,
    LL_ROLLOUT_STEPS, LL_N_EPOCHS, LL_N_MINIBATCHES, LR_LOW, ENT_COEF_LL,
    META_ROLLOUT_STEPS, META_N_EPOCHS, META_N_MINIBATCHES, LR_META, ENT_COEF_META,
    CHECKPOINT_EVERY,
)
from rl_agent.models import LowLevelPolicy, MetaPolicy
from rl_agent.ppo import PPOTrainer, RolloutBuffer


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

def _worker_fn(worker_id: int, conn, seed: int) -> None:
    """Runs in a subprocess: receives actions, sends (obs, reward, done, info)."""
    # Workers run pure CPU — disable CUDA before any torch import to avoid
    # CUDA re-init issues in spawned/forked processes.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["FLOW_HEADLESS"] = "1"
    import sys, os as _os
    _REPO_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)
    from rl_agent.env import FlowGameEnv

    env = FlowGameEnv(seed=seed + worker_id)
    obs = env.reset()
    conn.send(("reset_done", obs))

    while True:
        msg, payload = conn.recv()
        if msg == "step":
            action, target_arch = payload
            env.set_target(int(target_arch))
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                obs = env.reset()
            conn.send(("step_done", (obs, reward, done, info)))

        elif msg == "set_target":
            env.set_target(int(payload))
            conn.send(("ok", None))

        elif msg == "get_meta_obs":
            score_delta_K = payload
            meta_obs = env.get_meta_obs(score_delta_K)
            conn.send(("meta_obs", meta_obs))

        elif msg == "reset":
            obs = env.reset()
            conn.send(("reset_done", obs))

        elif msg == "close":
            conn.close()
            return


# ---------------------------------------------------------------------------
# VecEnv-like coordinator
# ---------------------------------------------------------------------------

class ParallelEnvs:
    """Manages N worker processes (spawned, not forked — safe with CUDA)."""

    def __init__(self, n_envs: int, base_seed: int = 0) -> None:
        self.n_envs = n_envs
        self._parent_conns = []
        self._procs: List[Process] = []
        ctx = mp.get_context("spawn")
        for i in range(n_envs):
            parent_conn, child_conn = ctx.Pipe()
            p = ctx.Process(
                target=_worker_fn,
                args=(i, child_conn, base_seed),
                daemon=True,
            )
            p.start()
            self._parent_conns.append(parent_conn)
            self._procs.append(p)
        # Wait for all workers to initialise.
        self._obs = np.stack([
            self._recv(i, "reset_done") for i in range(n_envs)
        ])

    def _send(self, i: int, msg: str, payload=None) -> None:
        self._parent_conns[i].send((msg, payload))

    def _recv(self, i: int, expected_msg: str):
        msg, payload = self._parent_conns[i].recv()
        assert msg == expected_msg, f"worker {i}: expected {expected_msg}, got {msg}"
        return payload

    def reset_all(self) -> np.ndarray:
        for i in range(self.n_envs):
            self._send(i, "reset")
        self._obs = np.stack([self._recv(i, "reset_done") for i in range(self.n_envs)])
        return self._obs

    def step(
        self,
        actions: np.ndarray,            # [N, 3]
        target_archs: np.ndarray,       # [N] int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        for i in range(self.n_envs):
            self._send(i, "step", (actions[i], target_archs[i]))
        results = [self._recv(i, "step_done") for i in range(self.n_envs)]
        obs_list, rew_list, done_list, info_list = zip(*results)
        self._obs = np.stack(obs_list)
        return (
            self._obs,
            np.array(rew_list, dtype=np.float32),
            np.array(done_list, dtype=np.float32),
            list(info_list),
        )

    def get_meta_obs(self, score_deltas_K: np.ndarray) -> np.ndarray:
        for i in range(self.n_envs):
            self._send(i, "get_meta_obs", float(score_deltas_K[i]))
        return np.stack([self._recv(i, "meta_obs") for i in range(self.n_envs)])

    @property
    def obs(self) -> np.ndarray:
        return self._obs

    def close(self) -> None:
        for i in range(self.n_envs):
            try:
                self._send(i, "close")
            except (BrokenPipeError, OSError):
                pass
        for p in self._procs:
            p.join(timeout=3)
            if p.is_alive():
                p.terminate()


# ---------------------------------------------------------------------------
# Hierarchical trainer
# ---------------------------------------------------------------------------

class HierarchicalTrainer:
    """Trains both policies concurrently."""

    def __init__(
        self,
        n_envs: int = NUM_ENVS,
        total_timesteps: int = 10_000_000,
        checkpoint_dir: str = "rl_agent/checkpoints",
        resume: Optional[str] = None,
        seed: int = 0,
    ) -> None:
        self.n_envs           = n_envs
        self.total_timesteps  = total_timesteps
        self.checkpoint_dir   = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Policies (on GPU).
        self.ll_policy   = LowLevelPolicy().to(DEVICE)
        self.meta_policy = MetaPolicy().to(DEVICE)

        # PPO trainers.
        self.ll_trainer = PPOTrainer(
            self.ll_policy, lr=LR_LOW,
            n_epochs=LL_N_EPOCHS, n_minibatches=LL_N_MINIBATCHES,
            ent_coef=ENT_COEF_LL, device=DEVICE,
        )
        self.meta_trainer = PPOTrainer(
            self.meta_policy, lr=LR_META,
            n_epochs=META_N_EPOCHS, n_minibatches=META_N_MINIBATCHES,
            ent_coef=ENT_COEF_META, device=DEVICE,
        )

        # Rollout buffers (one per policy; covers all envs).
        ll_cap   = LL_ROLLOUT_STEPS * n_envs
        meta_cap = (META_ROLLOUT_STEPS // OPTION_PERIOD_K) * n_envs
        self.ll_buf = RolloutBuffer(
            capacity=ll_cap, obs_dim=OBS_DIM, act_dim=3, extra_dim=1
        )
        self.meta_buf = RolloutBuffer(
            capacity=max(meta_cap, n_envs * 8),
            obs_dim=META_OBS_DIM, act_dim=1,
        )

        # Per-env state.
        self.target_archs      = np.zeros(n_envs, dtype=np.int32)
        self.steps_since_meta  = np.zeros(n_envs, dtype=np.int32)
        self.score_accum_K     = np.zeros(n_envs, dtype=np.float32)
        self.meta_obs_at_start = np.zeros((n_envs, META_OBS_DIM), dtype=np.float32)
        self.meta_lp_at_start  = np.zeros(n_envs, dtype=np.float32)
        self.meta_val_at_start = np.zeros(n_envs, dtype=np.float32)

        self.global_step = 0
        self.episodes    = 0
        self.t_start     = time.time()

        # Parallel envs.
        self.envs = ParallelEnvs(n_envs, base_seed=seed)

        if resume is not None:
            self._load_checkpoint(resume)

        # Initialise meta decisions.
        self._meta_step_all(self.envs.obs)

    # ------------------------------------------------------------------

    def train(self) -> None:
        obs = self.envs.obs.copy()

        while self.global_step < self.total_timesteps:
            # --- Collect LL_ROLLOUT_STEPS per env -------------------------
            self.ll_buf.reset()
            ll_last_obs   = np.zeros_like(obs)
            ll_last_dones = np.zeros(self.n_envs, dtype=np.float32)

            for _ in range(LL_ROLLOUT_STEPS):
                obs_t = torch.from_numpy(obs).to(DEVICE)
                is_shop_t = torch.from_numpy(
                    obs[:, 18].astype(np.float32) > 0.5    # obs[18] = is_shop flag (idx 5+8+4+5=22... wait)
                ).to(DEVICE)
                # is_shop is at obs index: 5+8+4+5+5 = 27, then is_shop is obs[27]
                is_shop_t = torch.from_numpy(obs[:, 27] > 0.5).to(DEVICE)

                with torch.no_grad():
                    actions_t, lp_t, _, val_t = self.ll_policy.get_action_and_value(
                        obs_t, is_shop_t
                    )

                actions_np = actions_t.cpu().numpy()
                lp_np      = lp_t.cpu().float().numpy()
                val_np     = val_t.cpu().float().numpy()

                next_obs, rewards, dones, infos = self.envs.step(
                    actions_np, self.target_archs
                )

                # Add to LL buffer.
                for i in range(self.n_envs):
                    self.ll_buf.add(
                        obs=obs[i],
                        action=actions_np[i],
                        reward=rewards[i],
                        value=val_np[i],
                        log_prob=lp_np[i],
                        done=dones[i],
                        extra=np.array([float(obs[i, 27] > 0.5)]),
                    )
                    self.score_accum_K[i] += rewards[i]
                    self.steps_since_meta[i] += 1
                    if dones[i]:
                        self.episodes += 1

                    # Meta option timeout.
                    if self.steps_since_meta[i] >= OPTION_PERIOD_K:
                        self._finish_meta_option(i, next_obs[i], dones[i])
                        self._start_meta_option(i, next_obs[i])

                obs = next_obs
                self.global_step += self.n_envs
                ll_last_obs   = next_obs
                ll_last_dones = dones

                if self.ll_buf.is_full():
                    break

            # --- LL PPO update -------------------------------------------
            obs_t   = torch.from_numpy(ll_last_obs).to(DEVICE)
            is_shop = torch.from_numpy(ll_last_obs[:, 27] > 0.5).to(DEVICE)
            with torch.no_grad():
                _, _, _, last_val_t = self.ll_policy.get_action_and_value(
                    obs_t, is_shop
                )
            last_val_mean  = last_val_t.mean().item()
            last_done_mean = ll_last_dones.mean()

            ll_metrics = self.ll_trainer.update(
                self.ll_buf, last_val_mean, last_done_mean, is_low_level=True
            )

            # --- Meta PPO update (when buffer has enough entries) ---------
            if self.meta_buf.ptr >= self.meta_buf.capacity // 2:
                # Bootstrap last meta value.
                meta_obs_t = torch.from_numpy(
                    self.envs.get_meta_obs(self.score_accum_K)
                ).to(DEVICE)
                with torch.no_grad():
                    _, meta_last_val = self.meta_policy(meta_obs_t)
                meta_last_val_mean = meta_last_val.mean().item()

                meta_metrics = self.meta_trainer.update(
                    self.meta_buf, meta_last_val_mean, 0.0, is_low_level=False
                )
                self.meta_buf.reset()
            else:
                meta_metrics = {}

            # --- Logging --------------------------------------------------
            elapsed = time.time() - self.t_start
            sps     = self.global_step / max(elapsed, 1e-6)
            print(
                f"step={self.global_step:>9,}  eps={self.episodes:>5}  "
                f"sps={sps:>6,.0f}  "
                f"ll_pg={ll_metrics['pg_loss']:+.4f}  "
                f"ll_vf={ll_metrics['vf_loss']:.4f}  "
                f"ll_ent={ll_metrics['ent_loss']:.4f}"
                + (f"  meta_pg={meta_metrics.get('pg_loss', 0):+.4f}" if meta_metrics else "")
            )

            # --- Checkpoint -----------------------------------------------
            if self.global_step % CHECKPOINT_EVERY < (self.n_envs * LL_ROLLOUT_STEPS):
                self._save_checkpoint()

        self._save_checkpoint(final=True)
        self.envs.close()
        print("Training complete.")

    # ------------------------------------------------------------------

    def _meta_step_all(self, obs: np.ndarray) -> None:
        """Initial meta decision for all envs."""
        meta_obs = self.envs.get_meta_obs(np.zeros(self.n_envs))
        for i in range(self.n_envs):
            self._start_meta_option(i, obs[i], meta_obs_override=meta_obs[i])

    def _start_meta_option(
        self, env_idx: int, ll_obs: np.ndarray,
        meta_obs_override: Optional[np.ndarray] = None
    ) -> None:
        if meta_obs_override is not None:
            meta_obs = meta_obs_override
        else:
            self.envs._send(env_idx, "get_meta_obs", float(self.score_accum_K[env_idx]))
            meta_obs = self.envs._recv(env_idx, "meta_obs")

        meta_obs_t = torch.from_numpy(meta_obs[None]).to(DEVICE)
        with torch.no_grad():
            arch_t, lp_t, _, val_t = self.meta_policy.get_action_and_value(meta_obs_t)
        arch_int = int(arch_t[0].item())
        self.target_archs[env_idx]      = arch_int
        self.meta_obs_at_start[env_idx] = meta_obs
        self.meta_lp_at_start[env_idx]  = float(lp_t[0].item())
        self.meta_val_at_start[env_idx] = float(val_t[0].item())
        self.steps_since_meta[env_idx]  = 0
        self.score_accum_K[env_idx]     = 0.0

    def _finish_meta_option(
        self, env_idx: int, next_ll_obs: np.ndarray, done: float
    ) -> None:
        meta_reward = float(self.score_accum_K[env_idx])
        self.meta_buf.add(
            obs      = self.meta_obs_at_start[env_idx],
            action   = np.array([self.target_archs[env_idx]], dtype=np.int64),
            reward   = meta_reward,
            value    = self.meta_val_at_start[env_idx],
            log_prob = self.meta_lp_at_start[env_idx],
            done     = done,
        )

    # ------------------------------------------------------------------

    def _save_checkpoint(self, final: bool = False) -> None:
        tag  = "final" if final else f"step{self.global_step}"
        path = os.path.join(self.checkpoint_dir, f"checkpoint_{tag}.pt")
        torch.save({
            "global_step":      self.global_step,
            "ll_policy":        self.ll_policy.state_dict(),
            "meta_policy":      self.meta_policy.state_dict(),
            "ll_optimizer":     self.ll_trainer.optimizer.state_dict(),
            "meta_optimizer":   self.meta_trainer.optimizer.state_dict(),
        }, path)
        print(f"[checkpoint] saved {path}")

    def _load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=DEVICE)
        self.ll_policy.load_state_dict(ckpt["ll_policy"])
        self.meta_policy.load_state_dict(ckpt["meta_policy"])
        self.ll_trainer.optimizer.load_state_dict(ckpt["ll_optimizer"])
        self.meta_trainer.optimizer.load_state_dict(ckpt["meta_optimizer"])
        self.global_step = ckpt.get("global_step", 0)
        print(f"[checkpoint] resumed from {path} at step {self.global_step}")
