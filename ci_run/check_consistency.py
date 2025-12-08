#!/usr/bin/env python3
import subprocess
import time
import sys
import signal
import atexit
import os
import json
import argparse
import yaml
from pathlib import Path
import shutil
import h5py
import numpy as np

LW_BENCHHUB_ROOT = Path(__file__).parent.parent.absolute()
DATASET_PATH = f"{LW_BENCHHUB_ROOT}/datasets"


def cleanup_process(process):
    """cleanup process function"""
    if process and process.poll() is None:
        print("terminating teleop process and its children...")
        try:
            try:
                pgid = os.getpgid(process.pid)
                print(f"terminating process group {pgid}...")
                os.killpg(pgid, signal.SIGTERM)

                process.wait(timeout=15)
                print("process group terminated normally")
            except (OSError, subprocess.TimeoutExpired):
                print("process group termination failed, trying to terminate main process...")
                process.terminate()
                process.wait(timeout=15)
                print("main process terminated")
        except subprocess.TimeoutExpired:
            print("force killing process group...")
            try:
                pgid = os.getpgid(process.pid)
                os.killpg(pgid, signal.SIGKILL)
                process.wait(timeout=2)
                print("process group force killed")
            except (OSError, subprocess.TimeoutExpired):
                process.kill()
                try:
                    process.wait(timeout=2)
                    print("main process force killed")
                except subprocess.TimeoutExpired:
                    print("warning: unable to terminate main process")


def update_config_task(task_name, layout):
    """Update the task field in check_consistency.yml"""
    try:
        # Read the original check_consistency.yml
        config_path = f"{LW_BENCHHUB_ROOT}/configs/data_collection/teleop/check_consistency.yml"
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        # Store original task for restoration
        original_task = config.get('task', 'OpenDishwasher')

        # Update the task field
        config['task'] = task_name
        if layout:
            config['layout'] = layout
        # Write back to the file
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"Updated check_consistency.yml with task: {task_name}")
        return original_task

    except Exception as e:
        print(f"Error updating config: {e}")
        return None


def send_keyboard_input(process, input_text):
    """Send keyboard input to the process"""
    try:
        if process and process.poll() is None:
            process.stdin.write(input_text + '\n')
            process.stdin.flush()
            # pyautogui.press(input_text)
            print(f"Sent keyboard input: {input_text}")
            return True
        else:
            print("Process is not running, cannot send input")
            return False
    except Exception as e:
        print(f"Error sending keyboard input: {e}")
        return False


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Monitor teleop_main.py execution')
    parser.add_argument('--task', type=str, default='OpenDishwasher',
                        help='Task name to use in the config (default: OpenDishwasher)')
    parser.add_argument('--layout', type=str, default=None,
                        help='Floorplan to use in the config')
    return parser.parse_args()


def run_teleop(rerun_count):
    # Parse command line arguments
    args = parse_arguments()

    print(f"starting teleop_main.py monitoring with task: {args.task}...")

    process = None
    original_task = None
    test_result = {
        "success": False,
        "desc": "init desc",
        "error": "none",
    }

    try:
        if rerun_count == 0:
            # Update the config file with the specified task
            original_task = update_config_task(args.task, args.layout)
            if original_task is None:
                print("Failed to update config file")
                return False

            print(f"\n[RUN] Start teleop:")

            process = subprocess.Popen(
                ["python3", "-u", f"{LW_BENCHHUB_ROOT}/lw_benchhub/scripts/teleop/teleop_main.py", "--task_config=check_consistency", "--headless"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                preexec_fn=os.setsid
            )

            atexit.register(cleanup_process, process)

            def signal_handler(signum, frame):  # pylint: disable=unused-argument
                print(f"received signal {signum}, cleaning up...")
                cleanup_process(process)
                sys.exit(0)

            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            signal.signal(signal.SIGHUP, signal_handler)

            teleop_begins_detected = False
            traceback_lines = []

            print("wait for 'Start Recording' output...")

            monitor_start_time = time.time()
            timeout = 300

            # --------------------------First run-----------------
            in_traceback = False
            for line in iter(process.stdout.readline, ''):
                if line:
                    line = line.strip()
                    print(f"[TELEOP] {line}")

                    if "Traceback (most recent call last):" in line:
                        in_traceback = True
                        traceback_lines = [line]
                    elif in_traceback:
                        if line == "" or ("ms]" in line):
                            in_traceback = False
                            if traceback_lines:
                                test_result["error"] = traceback_lines
                                last_three_lines = traceback_lines[-3:] if len(traceback_lines) >= 3 else traceback_lines
                                test_result["desc"] = "\n".join(last_three_lines)
                        else:
                            traceback_lines.append(line)

                    if "Start Recording" in line and not teleop_begins_detected:
                        teleop_begins_detected = True
                        print("Detect keyword 'Start Recording', start to monitor...")
                        break

                    if "Starting teleoperation" in line:
                        print("Detect keyword 'Starting teleoperation', teleop_main is ready for input...")
                        time.sleep(8)
                        send_keyboard_input(process, "b")

            if process.poll() is not None:
                print("Process exited during main loop, reading remaining output...")
                remaining_output = process.stdout.read()
                if remaining_output:
                    print(f"[REMAINING OUTPUT] {remaining_output}")
                    for line in remaining_output.split('\n'):
                        if line.strip():
                            line = line.strip()
                            print(f"[TELEOP] {line}")
                            if "Traceback (most recent call last):" in line:
                                in_traceback = True
                                traceback_lines = [line]
                            elif in_traceback:
                                if line == "" or ("ms]" in line):
                                    in_traceback = False
                                    if traceback_lines:
                                        test_result["error"] = traceback_lines
                                        last_three_lines = traceback_lines[-3:] if len(traceback_lines) >= 3 else traceback_lines
                                        test_result["desc"] = "\n".join(last_three_lines)
                                else:
                                    traceback_lines.append(line)

            if in_traceback and traceback_lines:
                print("Found incomplete traceback after main loop, saving it...")
                test_result["error"] = traceback_lines
                last_three_lines = traceback_lines[-3:] if len(traceback_lines) >= 3 else traceback_lines
                test_result["desc"] = "\n".join(last_three_lines)

            if not teleop_begins_detected:
                if process.poll() is None:
                    if time.time() - monitor_start_time > timeout:
                        if traceback_lines:
                            test_result["success"] = False
                            test_result["desc"] = "Start over 5 minutes but with errors"
                        else:
                            test_result["success"] = True
                            test_result["desc"] = "Start over 5 minutes"
                        test_result["error"] = traceback_lines
                        cleanup_process(process)
                        return test_result["success"]
                else:
                    test_result["success"] = False
                    if not traceback_lines:
                        test_result["desc"] = "not detect Start keyword, teleop did not start"
                    else:
                        last_three_lines = traceback_lines[-3:] if len(traceback_lines) >= 3 else traceback_lines
                        test_result["desc"] = "\n".join(last_three_lines)
                    test_result["error"] = traceback_lines
                return False

            wait_duration = 30
            print(f"[STEP] Wait {wait_duration}s before first save (with monitoring)...")

            wait_end_time = time.time() + wait_duration
            process_alive = True
            while time.time() < wait_end_time:
                if process.poll() is not None:
                    print(f"[ERROR] Process terminated unexpectedly during the first wait. Exit code: {process.returncode}")
                    test_result["success"] = False
                    test_result["desc"] = f"Process terminated unexpectedly during the first wait.{wait_duration}"
                    process_alive = False
                    break
                time.sleep(0.1)

            if process_alive:
                print("[STEP] Sending 't' to save demo_0...")
                send_keyboard_input(process, "t")
                time.sleep(5)
            else:
                return False

            # --------------------------X Reset-----------------
            print("[STEP] Sending 'x' to reset...")
            send_keyboard_input(process, "x")
            print("[STEP] Wait 15s for X reset to complete...")
            time.sleep(15)
            print("[STEP] X Reset done.")

            send_keyboard_input(process, "b")

            print(f"[STEP] Wait {wait_duration}s before second save (with monitoring)...")
            process_alive = True
            wait_end_time = time.time() + wait_duration
            while time.time() < wait_end_time:
                if process.poll() is not None:
                    print(f"[ERROR] Process terminated unexpectedly during the second wait. Exit code: {process.returncode}")
                    test_result["success"] = False
                    test_result["desc"] = f"Process terminated unexpectedly during the second wait.{wait_duration}"
                    process_alive = False
                    break
                time.sleep(0.1)

            if process_alive:
                print("[STEP] Sending 't' to save demo_1...")
                send_keyboard_input(process, "t")
                time.sleep(5)
            else:
                return False

            # --------------------------R Reset-----------------
            print("[STEP] Sending 'r' to start new phase...")
            send_keyboard_input(process, "r")
            print("[STEP] Wait 15s for R reset to complete...")
            time.sleep(15)
            print("[STEP] R reset done.")

            send_keyboard_input(process, "b")

            print(f"[STEP] Wait {wait_duration}s before third save (with monitoring)...")
            process_alive = True
            wait_end_time = time.time() + wait_duration
            while time.time() < wait_end_time:
                if process.poll() is not None:
                    print(f"[ERROR] Process terminated unexpectedly during the third wait. Exit code: {process.returncode}")
                    test_result["success"] = False
                    test_result["desc"] = f"Process terminated unexpectedly during the third wait.{wait_duration}"
                    process_alive = False
                    break
                time.sleep(0.1)

            if process_alive:
                print("[STEP] Sending 't' to save demo_2...")
                send_keyboard_input(process, "t")
                time.sleep(5)
            else:
                return False

            print("[STEP] Terminating teleop_main ...")
            print(f"[DONE] teleop_main run {rerun_count} finished.\n")

        else:
            print(f"\n[RUN] Start replay:")
            process = subprocess.Popen(
                ["python3", "-u", f"{LW_BENCHHUB_ROOT}/lw_benchhub/scripts/teleop/replay_action_demo.py", f"--dataset_file={DATASET_PATH}/dataset.hdf5", "--replay_mode=action", "--device=cpu", "--record", "--headless", "--select_episodes", "0", "1", "2"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1,
                preexec_fn=os.setsid
            )
            for line in iter(process.stdout.readline, ''):
                if "Finished replaying" in line:
                    print("Detect keyword 'Finished replaying', renaming dataset...")
                    time.sleep(3)
            print("[STEP] Terminating replay ...")
            print(f"[DONE] replay finished.\n")

        if "success" not in test_result:
            test_result["success"] = True
            test_result["desc"] = "Teleop run and save completed successfully."

    except Exception as e:
        test_result["error"] = f"script error: {str(e)}"
        test_result["desc"] = "script error"
        if process:
            cleanup_process(process)
        return False
    finally:
        if process and process.poll() is None:
            print("[FINALLY] Cleaning up the process...")
            cleanup_process(process)


def main():
    rerun_total = 2

    for rerun_count in range(rerun_total):
        run_teleop(rerun_count)

    print("[DONE] All runs completed successfully.")


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8, np.uint16,
                            np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def generate_result_json(test_result):
    try:
        os.makedirs(f"/output", exist_ok=True)
        with open(f"/output/result.json", "w", encoding="utf-8") as fp:
            json.dump(test_result, fp, ensure_ascii=False, indent=2, cls=NumpyEncoder)
        print(f"Generated result.json at /output/result.json")
    except Exception as e:
        print(f"fail to create result.json: {e}")


def check_dataset_consistency():
    """
    Compare dataset.hdf5 and dataset_action_replay_record.hdf5
    for /data/demo_{0,1,2}/{actions, joint_targets/joint_pos_target, states/articulation/robot/joint_position}.
    Report which paths differ, how many differences, how many rows are compared,
    and show 10 sample differences.
    """
    dataset_files = [
        f"{DATASET_PATH}/dataset.hdf5",
        f"{DATASET_PATH}/dataset_action_replay_record.hdf5"
    ]

    demos = ["demo_0", "demo_1", "demo_2"]
    demo_tags = {
        "demo_0": "first run",
        "demo_1": "reset_env_instance_keep_placement",
        "demo_2": "reset_recording_instance",
    }

    check_subpaths = [
        "actions",
        "joint_targets/joint_pos_target",
        "states/articulation/robot/joint_position"
    ]

    test_result = {"success": True, "desc": "", "error": ""}
    precision = 1e-8
    max_rows = 1000

    for fpath in dataset_files:
        if not os.path.exists(fpath):
            print(f"[ERROR] Missing file: {fpath}")
            test_result["success"] = False
            test_result["error"] = f"missing file: {fpath}"
            test_result["desc"] = "Required dataset files not found."
            return test_result

    print("[CHECK] Start comparing dataset.hdf5 <-> dataset_action_replay_record.hdf5 ...")

    try:
        f1 = h5py.File(dataset_files[0], "r")
        f2 = h5py.File(dataset_files[1], "r")
        all_ok = True
        inconsistencies = []

        for demo_name in demos:
            tag = demo_tags.get(demo_name, "")
            print(f"\n[DEMO] Comparing {demo_name} ({tag}) ...")

            for subpath in check_subpaths:
                h5_path = f"/data/{demo_name}/{subpath}"

                if h5_path not in f1 or h5_path not in f2:
                    print(f"[WARN] Path {h5_path} not found in one of the datasets, skipping.")
                    continue

                try:
                    data1 = f1[h5_path][()]
                    data2 = f2[h5_path][()]

                    min_rows = min(data1.shape[0], data2.shape[0], max_rows)
                    data1 = data1[:min_rows]
                    data2 = data2[:min_rows]

                    if subpath == check_subpaths[0]:
                        print("\n[INFO] {0:<65} | {1:<40} | {2:^12} | {3}".format("HDF5 Path", "Tag", "Rows", "Status"))
                        print("[INFO] " + "-" * 140)

                    if not np.allclose(data1, data2, atol=precision, equal_nan=True):
                        all_ok = False
                        diff_mask = ~np.isclose(data1, data2, atol=precision, equal_nan=True)
                        diff_indices = np.argwhere(diff_mask)
                        diff_samples = []

                        for idx in diff_indices[:10]:
                            idx_tuple = tuple(idx)
                            diff_samples.append({
                                "index": idx_tuple,
                                "file1_value": float(data1[idx_tuple]),
                                "file2_value": float(data2[idx_tuple])
                            })

                        inconsistencies.append({
                            "demo": demo_name,
                            "path": subpath,
                            "rows_compared": int(min_rows),
                            "diff_count": int(len(diff_indices)),
                            "diff_samples": diff_samples
                        })

                        print("[DIFF] {0:<65} | {1:<40} | {2:^12} | {3}".format(
                            f"/data/{demo_name}/{subpath}",
                            f"({tag})",
                            str(min_rows),
                            f"{len(diff_indices)} differences"
                        ))

                        for ex in diff_samples:
                            print(f"        index {ex['index']}: {ex['file1_value']} != {ex['file2_value']}")
                    else:
                        print("[OK]   {0:<65} | {1:<40} | {2:^12} | Consistent".format(
                            f"/data/{demo_name}/{subpath}",
                            f"({tag})",
                            str(min_rows)
                        ))
                except Exception as e:
                    all_ok = False
                    print(f"[ERROR] Compare failed at {h5_path}: {e}")
                    inconsistencies.append({
                        "demo": demo_name,
                        "tag": tag,
                        "path": subpath,
                        "error": str(e)
                    })

        f1.close()
        f2.close()

        if all_ok:
            print("\n[CHECK] All compared paths are consistent.")
            test_result["success"] = True
            test_result["desc"] = "All compared demos and paths are consistent."
            test_result["error"] = "none"
        else:
            print("\n[CHECK] Inconsistencies detected!")
            test_result["success"] = False
            test_result["desc"] = "Differences found in datasets."
            test_result["error"] = inconsistencies

    except Exception as e:
        print(f"[CHECK] Fatal error: {e}")
        test_result["success"] = False
        test_result["desc"] = "Error during dataset comparison."
        test_result["error"] = str(e)

    generate_result_json(test_result)
    return test_result


if __name__ == "__main__":
    main()
    print("\n[POST] Starting dataset consistency check...")
    check_dataset_consistency()
