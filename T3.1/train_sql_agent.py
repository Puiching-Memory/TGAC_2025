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
        "train_files": "data/train_spider.parquet",
        "val_files": "data/test_dev_500.parquet",
        "train_batch_size": 32,
        "max_prompt_length": 4096,
        "max_response_length": 2048,
        "truncation": "error",
    },
    "actor_rollout_ref": {
        "rollout": {
            "tensor_model_parallel_size": 1,
            "n": 4,
            "log_prob_micro_batch_size_per_gpu": 4,
            "multi_turn": {"format": "hermes"},
            "name": "vllm",
            "gpu_memory_utilization": 0.8,
            "engine_kwargs": {
                "vllm": {
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "hermes",
                }
            },
        },
        "actor": {
            "ppo_mini_batch_size": 32,
            "ppo_micro_batch_size_per_gpu": 4,
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
            "path": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
            "use_remove_padding": True,
            "enable_gradient_checkpointing": True,
        },
    },
    "trainer": {
        "n_gpus_per_node": 1,
        "val_before_train": True,
        "critic_warmup": 0,
        "logger": ["console", "wandb"],
        "project_name": "AgentLightning",
        "experiment_name": "spider",
        "nnodes": 1,
        "test_freq": 32,
        "total_epochs": 2,
    },
}

def train(config: Dict[str, Any], active_agent: Optional[str]) -> None:
    """Train the SQL agent with the given configuration."""

    agent = LitSQLAgent()
    algorithm = agl.VERL(config)
    trainer = agl.Trainer(n_runners=10, algorithm=algorithm, adapter={"agent_match": active_agent})
    print("Adapter agent match acknowledged:", trainer.adapter.agent_match)  # type: ignore

    train_data = pd.read_parquet(config["data"]["train_files"]).to_dict(orient="records")  # type: ignore
    val_data = pd.read_parquet(config["data"]["val_files"]).to_dict(orient="records")  # type: ignore
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