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
import os

import matplotlib.pyplot as plt
import yaml


def parse_metric(metric_data):
    """
    Supports two formats:
    1. dict: {"0": "0.0JS:0", "1": "12.5JS:0", ...}
    2. list: [0.0, 12.5, ...]
    """
    values = []
    if isinstance(metric_data, dict):
        for k in sorted(metric_data.keys(), key=lambda x: int(x)):
            v = metric_data[k]
            if isinstance(v, str) and "JS:" in v:
                v = v.split("JS:")[0]
            values.append(float(v))
    elif isinstance(metric_data, list):
        for v in metric_data:
            if isinstance(v, str) and "JS:" in v:
                v = v.split("JS:")[0]
            values.append(float(v))
    else:
        raise ValueError(f"Unsupported data type: {type(metric_data)}")
    return values


def plot_metrics(config_path):
    # Read YAML configuration
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    json_path = cfg.get("json_path")
    record = cfg.get("record", False)

    if json_path is None:
        raise ValueError("YAML must have 'json_path' field")

    # Read JSON data
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    clipping = data["clipping"]

    metrics_to_plot = ["force", "mean_force", "force_variance", "delta_force"]

    parsed = {m: parse_metric(clipping[m]) for m in metrics_to_plot}

    # Determine y-axis range
    if "ymin" in cfg and "ymax" in cfg:
        y_min, y_max = cfg["ymin"], cfg["ymax"]
    else:
        all_values = []
        for v in parsed.values():
            all_values.extend(v)
        y_min, y_max = min(all_values), max(all_values)

    save_path = os.path.splitext(json_path)[0] + f"_metrics_{y_min}_{y_max}.png"

    # Plot the metrics
    fig, axes = plt.subplots(len(metrics_to_plot), 1, figsize=(8, 12), sharex=True)

    for i, m in enumerate(metrics_to_plot):
        axes[i].plot(parsed[m], label=m, marker="o", markersize=2, linewidth=1)
        axes[i].set_ylabel(m)
        axes[i].set_ylim(y_min, y_max)
        axes[i].legend()
        axes[i].grid(axis="y", color="gray", linestyle="--", alpha=0.5)

    axes[-1].set_xlabel("Frame Index")
    plt.tight_layout()

    if record:
        plt.savefig(save_path, dpi=300)
        plt.show()
        print(f"Figure has been saved to: {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    # Pass the configuration file path
    plot_metrics("./lw_benchhub/core/checks/contact_force_plot_config.yaml")
