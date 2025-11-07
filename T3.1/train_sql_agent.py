# Copyright (c) Microsoft. All rights reserved.

"""Train an SQL agent on the Spider dataset using Agent-lightning.

This module provides a training script for SQL agents using Agent-lightning.
It exposes a single `qwen` configuration aligned with the base config.

Usage:
    python train_sql_agent.py [--active-agent NAME]

The script uses reinforcement learning with VERL framework
to train agents on the Spider dataset for text-to-SQL generation tasks.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from sql_agent import LitSQLAgent

import agentlightning as agl

RL_TRAINING_CONFIG: Dict[str, Any] = {
    "algorithm": {
        "adv_estimator": "grpo",
        "use_kl_in_reward": False,
    },
    "data": {
        "train_files": "../T3/data/final_dataset.json",
        "val_files": "../T3/data/final_dataset.json",
    "train_batch_size": 2, # Must remain >= n_gpus_per_node and divisible by it
        "max_prompt_length": 4096,
        "max_response_length": 2048,
        "truncation": "error",
    },
    "actor_rollout_ref": {
        "rollout": {
            "tensor_model_parallel_size": 2,
            "n": 1,
            "log_prob_micro_batch_size_per_gpu": 4,
            "multi_turn": {"format": "hermes"},
            "name": "vllm",
            "gpu_memory_utilization": 0.1,
        },
        "actor": {
            "ppo_mini_batch_size": 2, # Keep <= train_batch_size and >= n_gpus_per_node to avoid zero normalization
            "ppo_micro_batch_size_per_gpu": 1, # Must evenly divide normalized mini batch (= ppo_mini_batch_size / n_gpus_per_node)
            "optim": {"lr": 1e-6},
            "use_kl_loss": False,
            "kl_loss_coef": 0.0,
            "entropy_coeff": 0,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.3,
            "fsdp_config": {
                "param_offload": True,
                "optimizer_offload": True,
            },
        },
        "ref": {
            "log_prob_micro_batch_size_per_gpu": 8,
            "fsdp_config": {"param_offload": True},
        },
        "model": {
            "path": "XGenerationLab/XiYanSQL-QwenCoder-3B-2504",
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
        },
    },
    "trainer": {
        "n_gpus_per_node": 2,
        "val_before_train": True,
        "critic_warmup": 0,
        "logger": ["console"],
        "project_name": "AgentLightning",
        "experiment_name": "text2sql",
        "nnodes": 1,
        "test_freq": 32,
        "total_epochs": 2,
    },
}


def _resolve_path(path: str) -> str:
    candidate = Path(path)
    if candidate.is_absolute():
        return str(candidate)
    resolved = (Path(__file__).resolve().parent / candidate).resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Resolved path does not exist: {resolved}")
    return str(resolved)


def train(config: Dict[str, Any], active_agent: Optional[str]) -> None:
    """Train the SQL agent with the given configuration."""

    agent = LitSQLAgent()
    algorithm = agl.VERL(config)
    trainer = agl.Trainer(n_runners=1, algorithm=algorithm, adapter={"agent_match": active_agent})
    print("Adapter agent match acknowledged:", trainer.adapter.agent_match)  # type: ignore

    # Load data - support both parquet and JSON formats
    train_files = _resolve_path(config["data"]["train_files"])
    val_files = _resolve_path(config["data"]["val_files"])
    
    if train_files.endswith(".parquet"):
        train_df = pd.read_parquet(train_files)
        train_data = train_df.where(pd.notnull(train_df), None).to_dict(orient="records")  # type: ignore
    elif train_files.endswith(".json"):
        train_df = pd.read_json(train_files)
        train_data = train_df.where(pd.notnull(train_df), None).to_dict(orient="records")  # type: ignore
    else:
        raise ValueError(f"Unsupported file format: {train_files}")
    
    if val_files.endswith(".parquet"):
        val_df = pd.read_parquet(val_files)
        val_data = val_df.where(pd.notnull(val_df), None).to_dict(orient="records")  # type: ignore
    elif val_files.endswith(".json"):
        val_df = pd.read_json(val_files)
        val_data = val_df.where(pd.notnull(val_df), None).to_dict(orient="records")  # type: ignore
    else:
        raise ValueError(f"Unsupported file format: {val_files}")
    
    trainer.fit(agent, train_dataset=train_data, val_dataset=val_data)  # type: ignore


def main() -> None:
    """Main function to parse arguments and run training."""
    parser = argparse.ArgumentParser(
        description="Train an SQL agent on the Spider dataset using the default configuration"
    )

    parser.add_argument(
        "--active-agent", type=str, help="Override the active agent name (default: auto-generated based on config)"
    )

    args = parser.parse_args()

    # Get the configuration (single preset)
    config = deepcopy(RL_TRAINING_CONFIG)

    # Set active agent - use provided value or default based on config choice
    active_agent = args.active_agent

    print("Starting training with 'qwen' configuration...")
    print(f"Active agent: {active_agent}")

    train(config, active_agent)


if __name__ == "__main__":
    main()