import h5py
import numpy as np
import os
import argparse
import sys


def detect_motion_start(joint_positions, threshold=0.01, window_size=5):
    """
    Detect when robot starts moving based on joint position changes.

    Args:
        joint_positions: Joint position data (timesteps, num_joints)
        threshold: Threshold for joint position changes
        window_size: Sliding window size for smoothing

    Returns:
        start_idx: Index where motion starts
    """
    if len(joint_positions) < 2:
        return 0

    # Calculate joint position differences between consecutive timesteps
    joint_diff = np.abs(np.diff(joint_positions, axis=0))
    max_joint_change = np.max(joint_diff, axis=1)

    # Apply sliding window smoothing
    if len(max_joint_change) >= window_size:
        smoothed_change = np.convolve(max_joint_change, np.ones(window_size) / window_size, mode='valid')
        offset = window_size // 2
    else:
        smoothed_change = max_joint_change
        offset = 0

    # Find first point above threshold
    motion_indices = np.where(smoothed_change > threshold)[0]
    return motion_indices[0] + offset if len(motion_indices) > 0 else 0


def trim_static_data(demo_data, start_idx):
    """
    Trim static data, keeping data from start_idx onwards.

    Args:
        demo_data: HDF5 demo data group
        start_idx: Starting index to keep

    Returns:
        trimmed_data: Dictionary of trimmed data
    """
    trimmed_data = {}

    def copy_data(name, obj):
        if isinstance(obj, h5py.Dataset):
            if len(obj.shape) > 0 and obj.shape[0] > start_idx:
                trimmed_data[name] = obj[start_idx:]
            else:
                trimmed_data[name] = obj[:]
        elif isinstance(obj, h5py.Group):
            for sub_name, sub_obj in obj.items():
                copy_data(f"{name}/{sub_name}", sub_obj)

    for name, obj in demo_data.items():
        copy_data(name, obj)

    return trimmed_data


def save_trimmed_data(original_path, trimmed_data, start_idx):
    """
    Save trimmed data to new HDF5 file.

    Args:
        original_path: Original file path
        trimmed_data: Trimmed data dictionary
        start_idx: Starting index of trimming
    """
    dir_name = os.path.dirname(original_path)
    base_name = os.path.basename(original_path)
    name_without_ext = os.path.splitext(base_name)[0]
    new_path = os.path.join(dir_name, f"{name_without_ext}_trimmed.hdf5")

    with h5py.File(new_path, 'w') as new_f:
        data_group = new_f.create_group('data')
        demo_group = data_group.create_group('demo_0')

        for name, data in trimmed_data.items():
            demo_group.create_dataset(name, data=data)

        # Add metadata
        trimmed_length = len(list(trimmed_data.values())[0])
        demo_group.attrs['original_length'] = trimmed_length + start_idx
        demo_group.attrs['trimmed_start_idx'] = start_idx
        demo_group.attrs['trimmed_length'] = trimmed_length

    print(f"Trimmed data saved to: {new_path}")
    return new_path


def truncate_demo_data(demo_data, max_length):
    """
    Truncate demo data to fixed length.

    Args:
        demo_data: HDF5 demo data group
        max_length: Maximum length

    Returns:
        truncated_data: Dictionary of truncated data
    """
    truncated_data = {}

    def copy_truncated_data(name, obj):
        if isinstance(obj, h5py.Dataset):
            if len(obj.shape) > 0:
                actual_length = min(obj.shape[0], max_length)
                truncated_data[name] = obj[:actual_length]
            else:
                truncated_data[name] = obj[:]
        elif isinstance(obj, h5py.Group):
            for sub_name, sub_obj in obj.items():
                copy_truncated_data(f"{name}/{sub_name}", sub_obj)

    for name, obj in demo_data.items():
        copy_truncated_data(name, obj)

    return truncated_data


def save_truncated_data(original_path, all_truncated_data, max_length):
    """
    Save all truncated demo data to new HDF5 file.

    Args:
        original_path: Original file path
        all_truncated_data: Dictionary of all truncated demo data
        max_length: Maximum truncation length
    """
    dir_name = os.path.dirname(original_path)
    base_name = os.path.basename(original_path)
    name_without_ext = os.path.splitext(base_name)[0]
    new_path = os.path.join(dir_name, f"{name_without_ext}_truncated_{max_length}.hdf5")

    with h5py.File(new_path, 'w') as new_f:
        data_group = new_f.create_group('data')

        for demo_name, (truncated_data, original_length) in all_truncated_data.items():
            demo_group = data_group.create_group(demo_name)

            for name, data in truncated_data.items():
                demo_group.create_dataset(name, data=data)

            # Add metadata
            demo_group.attrs['original_length'] = original_length
            demo_group.attrs['truncated_length'] = max_length
            demo_group.attrs['actual_length'] = len(list(truncated_data.values())[0])

    print(f"Truncated data saved to: {new_path}")
    return new_path


def truncate_hdf5(hdf5_path, max_length):
    """
    Truncate all demos in HDF5 file to fixed length.

    Args:
        hdf5_path: HDF5 file path
        max_length: Maximum truncation length
    """
    with h5py.File(hdf5_path, 'r') as f:
        data_group = f["/data"]
        demo_keys = [key for key in data_group.keys() if key.startswith('demo_')]
        print(f"Found {len(demo_keys)} demos, truncating to max length: {max_length}")

        all_truncated_data = {}

        for demo_key in demo_keys:
            demo = data_group[demo_key]

            # Get original length from first time-series data
            original_length = 0
            for key in demo.keys():
                if hasattr(demo[key], 'shape') and len(demo[key].shape) > 0:
                    original_length = demo[key].shape[0]
                    break
                elif isinstance(demo[key], h5py.Group):
                    for sub_key in demo[key].keys():
                        if hasattr(demo[key][sub_key], 'shape') and len(demo[key][sub_key].shape) > 0:
                            original_length = demo[key][sub_key].shape[0]
                            break
                    if original_length > 0:
                        break

            truncated_data = truncate_demo_data(demo, max_length)
            all_truncated_data[demo_key] = (truncated_data, original_length)

            status = "truncated" if original_length > max_length else "kept"
            print(f"{demo_key}: {original_length} -> {min(original_length, max_length)} ({status})")

        if all_truncated_data:
            save_truncated_data(hdf5_path, all_truncated_data, max_length)
        else:
            print("No demo data processed")


def save_all_trimmed_data(original_path, all_trimmed_data):
    """
    Save all trimmed demo data to new HDF5 file.

    Args:
        original_path: Original file path
        all_trimmed_data: Dictionary of all trimmed demo data
    """
    dir_name = os.path.dirname(original_path)
    base_name = os.path.basename(original_path)
    name_without_ext = os.path.splitext(base_name)[0]
    new_path = os.path.join(dir_name, f"{name_without_ext}_all_trimmed.hdf5")

    with h5py.File(new_path, 'w') as new_f:
        data_group = new_f.create_group('data')

        for demo_name, (trimmed_data, start_idx, original_length) in all_trimmed_data.items():
            demo_group = data_group.create_group(demo_name)

            for name, data in trimmed_data.items():
                demo_group.create_dataset(name, data=data)

            # Add metadata
            demo_group.attrs['original_length'] = original_length
            demo_group.attrs['trimmed_start_idx'] = start_idx
            demo_group.attrs['trimmed_length'] = len(list(trimmed_data.values())[0])

    print(f"Trimmed data saved to: {new_path}")
    return new_path


def preprocess_hdf5(hdf5_path, motion_threshold=0.01, window_size=5):
    """
    Preprocess HDF5 file by removing static robot data.

    Args:
        hdf5_path: HDF5 file path
        motion_threshold: Motion detection threshold
        window_size: Sliding window size
    """
    with h5py.File(hdf5_path, 'r') as f:
        data_group = f["/data"]
        demo_keys = [key for key in data_group.keys() if key.startswith('demo_')]
        print(f"Found {len(demo_keys)} demos")

        all_trimmed_data = {}

        for demo_key in demo_keys:
            demo = data_group[demo_key]

            try:
                robot_joint_pos = demo["states/articulation/robot/joint_position"]

                # Detect motion start
                motion_start_idx = detect_motion_start(robot_joint_pos[:], motion_threshold, window_size)
                original_length = len(robot_joint_pos)

                if motion_start_idx > 0:
                    trimmed_data = trim_static_data(demo, motion_start_idx)
                    all_trimmed_data[demo_key] = (trimmed_data, motion_start_idx, original_length)
                    status = f"trimmed from {motion_start_idx}"
                else:
                    trimmed_data = trim_static_data(demo, 0)
                    all_trimmed_data[demo_key] = (trimmed_data, 0, original_length)
                    status = "no trimming needed"

                print(f"{demo_key}: {original_length} -> {original_length - motion_start_idx} ({status})")

            except KeyError as e:
                print(f"Error in {demo_key}: missing joint data {e}")
                continue
            except Exception as e:
                print(f"Error processing {demo_key}: {e}")
                continue

        if all_trimmed_data:
            save_all_trimmed_data(hdf5_path, all_trimmed_data)
        else:
            print("No demo data processed")


def downsample_hdf5(hdf5_path, ratio):
    """
    Downsample HDF5 file by a ratio. If ratio is 1/2, keep one and delete the following one, and so on.

    Args:
        hdf5_path: HDF5 file path
        ratio: Downsample ratio
    """
    with h5py.File(hdf5_path, 'r') as f:
        data_group = f["/data"]
        demo_keys = [key for key in data_group.keys() if key.startswith('demo_')]
        print(f"Found {len(demo_keys)} demos")

        all_downsampled_data = {}

        for demo_key in demo_keys:
            demo = data_group[demo_key]

            try:
                # Get original length from first time-series data
                original_length = 0
                for key in demo.keys():
                    if hasattr(demo[key], 'shape') and len(demo[key].shape) > 0:
                        original_length = demo[key].shape[0]
                        break
                    elif isinstance(demo[key], h5py.Group):
                        for sub_key in demo[key].keys():
                            if hasattr(demo[key][sub_key], 'shape') and len(demo[key][sub_key].shape) > 0:
                                original_length = demo[key][sub_key].shape[0]
                                break
                        if original_length > 0:
                            break

                # Calculate downsampled indices
                if original_length == 0:
                    # Handle empty datasets
                    indices = np.array([], dtype=int)
                    status = "empty dataset, skipping"
                elif ratio >= 1:
                    # If ratio >= 1, no downsampling needed
                    indices = np.arange(original_length)
                    status = "no downsampling needed"
                else:
                    # Calculate step size for downsampling
                    step = int(1 / ratio)
                    indices = np.arange(0, original_length, step)
                    status = f"downsampled by ratio {ratio} (step {step})"

                # Only process if we have valid indices
                if len(indices) > 0:
                    # Downsample the data
                    downsampled_data = downsample_demo_data(demo, indices)
                    all_downsampled_data[demo_key] = (downsampled_data, original_length, len(indices))
                else:
                    # Skip empty datasets
                    continue

                print(f"{demo_key}: {original_length} -> {len(indices)} ({status})")

            except Exception as e:
                print(f"Error processing {demo_key}: {e}")
                continue

        if all_downsampled_data:
            save_downsampled_data(hdf5_path, all_downsampled_data, ratio)
        else:
            print("No demo data processed")


def downsample_demo_data(demo_data, indices):
    """
    Downsample demo data using specified indices.

    Args:
        demo_data: HDF5 demo data group
        indices: Indices to keep for downsampling

    Returns:
        downsampled_data: Dictionary of downsampled data
    """
    downsampled_data = {}

    def copy_downsampled_data(name, obj):
        if isinstance(obj, h5py.Dataset):
            if len(obj.shape) > 0 and obj.shape[0] > 0:
                # Only downsample if the dataset has data and indices are valid
                if len(indices) > 0 and np.max(indices) < obj.shape[0]:
                    downsampled_data[name] = obj[indices]
                else:
                    # If indices are invalid, skip this dataset
                    return
            else:
                downsampled_data[name] = obj[:]
        elif isinstance(obj, h5py.Group):
            for sub_name, sub_obj in obj.items():
                copy_downsampled_data(f"{name}/{sub_name}", sub_obj)

    for name, obj in demo_data.items():
        copy_downsampled_data(name, obj)

    return downsampled_data


def save_downsampled_data(original_path, all_downsampled_data, ratio):
    """
    Save all downsampled demo data to new HDF5 file.

    Args:
        original_path: Original file path
        all_downsampled_data: Dictionary of all downsampled demo data
        ratio: Downsampling ratio
    """
    dir_name = os.path.dirname(original_path)
    base_name = os.path.basename(original_path)
    name_without_ext = os.path.splitext(base_name)[0]
    # Create a cleaner filename for the ratio
    if ratio < 1:
        # For ratios like 1/3, 1/4, etc., use fraction format
        if ratio == 1 / 2:
            ratio_str = "1_2"
        elif ratio == 1 / 3:
            ratio_str = "1_3"
        elif ratio == 1 / 4:
            ratio_str = "1_4"
        elif ratio == 1 / 5:
            ratio_str = "1_5"
        elif ratio == 1 / 10:
            ratio_str = "1_10"
        else:
            # For other ratios, use 2 decimal places
            ratio_str = f"{ratio:.2f}".replace(".", "_")
    else:
        ratio_str = str(ratio)

    new_path = os.path.join(dir_name, f"{name_without_ext}_downsampled_{ratio_str}.hdf5")

    with h5py.File(new_path, 'w') as new_f:
        data_group = new_f.create_group('data')

        for demo_name, (downsampled_data, original_length, downsampled_length) in all_downsampled_data.items():
            demo_group = data_group.create_group(demo_name)

            for name, data in downsampled_data.items():
                demo_group.create_dataset(name, data=data)

            # Add metadata
            demo_group.attrs['original_length'] = original_length
            demo_group.attrs['downsampled_length'] = downsampled_length
            demo_group.attrs['downsample_ratio'] = ratio

    print(f"Downsampled data saved to: {new_path}")
    return new_path


def print_usage_examples():
    """
    Print usage examples for the HDF5 processing tool.
    """
    print("HDF5 Processing Tool - Usage Examples")
    print("Mode 1: Remove static robot data")
    print("  python hdf5_utils.py dataset.hdf5 1")
    print("  python hdf5_utils.py dataset.hdf5 1 --motion_threshold 0.02 --window_size 10")
    print("\nMode 2: Truncate to fixed length")
    print("  python hdf5_utils.py dataset.hdf5 2")
    print("  python hdf5_utils.py dataset.hdf5 2 --max_length 800")
    print("\nMode 3: Trim static data then truncate")
    print("  python hdf5_utils.py dataset.hdf5 3 --max_length 800")
    print("\nMode 4: Downsample data")
    print("  python hdf5_utils.py dataset.hdf5 4")
    print("  python hdf5_utils.py dataset.hdf5 4 --ratio 0.25")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process HDF5 files for robot data.")
    parser.add_argument("--examples", action="store_true", help="Show usage examples")
    parser.add_argument("hdf5_file_path", nargs='?', help="Path to the HDF5 file to process.")
    parser.add_argument("mode", nargs='?', type=int, choices=[1, 2, 3, 4], help="Processing mode: 1=trim static, 2=truncate, 3=both, 4=downsample")
    parser.add_argument("--motion_threshold", type=float, default=0.01, help="Motion detection threshold for preprocess_hdf5.")
    parser.add_argument("--window_size", type=int, default=5, help="Sliding window size for preprocess_hdf5.")
    parser.add_argument("--max_length", type=int, default=1000, help="Maximum length for truncation.")
    parser.add_argument("--ratio", type=float, default=0.5, help="Downsample ratio for downsample_hdf5.")

    args = parser.parse_args()

    # Show examples if requested
    if args.examples:
        print_usage_examples()
        sys.exit(0)

    # Check if required arguments are provided
    if args.hdf5_file_path is None or args.mode is None:
        print("Error: Both hdf5_file_path and mode are required when not using --examples")
        print("Use --examples to see usage examples")
        sys.exit(1)

    # Check if file exists
    if not os.path.exists(args.hdf5_file_path):
        print(f"Error: File '{args.hdf5_file_path}' does not exist!")
        sys.exit(1)

    # Check if file is a valid HDF5 file
    try:
        with h5py.File(args.hdf5_file_path, 'r') as f:
            if '/data' not in f:
                print(f"Error: File '{args.hdf5_file_path}' does not contain '/data' group!")
                sys.exit(1)
    except Exception as e:
        print(f"Error: Cannot open '{args.hdf5_file_path}' as HDF5 file: {e}")
        sys.exit(1)

    print(f"Processing HDF5 file: {args.hdf5_file_path}")
    print(f"Mode: {args.mode}")

    try:
        if args.mode == 1:
            print("Mode 1: Remove static robot data")
            preprocess_hdf5(args.hdf5_file_path, motion_threshold=args.motion_threshold, window_size=args.window_size)

        elif args.mode == 2:
            print("Mode 2: Truncate to fixed length")
            truncate_hdf5(args.hdf5_file_path, args.max_length)

        elif args.mode == 3:
            print("Mode 3: Trim static data then truncate")
            preprocess_hdf5(args.hdf5_file_path, motion_threshold=args.motion_threshold, window_size=args.window_size)

            # Truncate the trimmed file
            dir_name = os.path.dirname(args.hdf5_file_path)
            base_name = os.path.basename(args.hdf5_file_path)
            name_without_ext = os.path.splitext(base_name)[0]
            trimmed_file_path = os.path.join(dir_name, f"{name_without_ext}_all_trimmed.hdf5")

            if os.path.exists(trimmed_file_path):
                truncate_hdf5(trimmed_file_path, args.max_length)
            else:
                print("Trimmed file not found, skipping truncation")

        elif args.mode == 4:
            print("Mode 4: Downsample data")
            downsample_hdf5(args.hdf5_file_path, args.ratio)

        print("Processing complete!")

    except Exception as e:
        print(f"Error during processing: {e}")
        sys.exit(1)
