# Copyright 2025 Lightwheel Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
from enum import IntEnum

from lw_benchhub.scripts.teleop.eval.motion_eval import MotionMetric


class MetricMode(IntEnum):
    DISABLED = 0
    SOFT = 1
    HARD = 2


# Maps QA metric class to dict with metric name, initializer, and kwargs to pass into the metric's respective
# @validate_episode calls
ALL_QA_METRICS = {
    "motion": {
        "cls": MotionMetric,
        "init": None,
        "mode": MetricMode.HARD,
        "warning": None,
        "task_whitelist": None,
        "task_blacklist": None,
        "validate_kwargs": dict(
            vel_limit=100
        ),
    }
}


def aggregate_episode_validation(task, episode_metrics):
    """
    Validates the given @episode_metrics

    Args:
        task (str): The name of the task whose QA metrics are being aggregated
        episode_metrics (dict): Keyword-mapped aggregated episode metrics

    Returns:
        2-tuple:
            - bool: Whether the validation succeeded or not (requires all metric validation checks to pass)
            - dict: Per-metric information
    """
    results = dict()
    metrics_list = []
    for metric_name in ALL_QA_METRICS.keys():
        metric_info = ALL_QA_METRICS[metric_name]
        # If disabled, skip
        if metric_info["mode"] == MetricMode.DISABLED:
            continue
        # If the task is not in the whitelist or is in the blacklist, skip
        if ((metric_info["task_whitelist"] is not None and task not in metric_info["task_whitelist"])
                or (metric_info["task_blacklist"] is not None and task in metric_info["task_blacklist"])):
            continue
        metrics_list.append(metric_name)
        results[metric_name] = metric_info["cls"].validate_episode(
            episode_metrics=episode_metrics,
            **metric_info["validate_kwargs"],
        )
        if not all(v["success"] for v in results[metric_name].values()) and metric_info["mode"] == MetricMode.SOFT:
            # Add warning feedback for manual QA
            results[metric_name]["warning"] = metric_info["warning"]

    # Passes if all metric validations pass
    success = all(v.get("success", True) for res in results.values() for v in res.values())
    return success, results, metrics_list


def evaluate_qa_metrics(all_episodes_metrics, results_fpath, task=""):
    all_success = True
    all_episodes_results = dict()
    for episode_id, episode_metrics in all_episodes_metrics.items():
        success, results, metrics_list = aggregate_episode_validation(task=task, episode_metrics=episode_metrics)

        if not success:
            all_episodes_results[episode_id] = {
                "success": success,
                "results": results,
            }
            all_success = False

        if not all_episodes_results.get("metrics", []):
            all_episodes_results["metrics"] = metrics_list

    if not all_success:
        print(f"Warning: Some QA metrics failed")
    all_episodes_results["success"] = all_success
    with open(results_fpath, "w+") as f:
        json.dump(all_episodes_results, f, indent=4)
