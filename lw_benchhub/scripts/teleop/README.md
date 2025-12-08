# Teleoperation Scripts

This directory contains teleoperation scripts for Isaac Lab environments with enhanced checkpoint and data preservation capabilities.

## Files

- `teleop_main.py` - Main teleoperation script with enhanced checkpoint system
- `teleop_utils.py` - Utility functions for teleoperation operations
- `replay_action_demo.py` - Script for replaying action demonstrations
- `replay_demos.py` - Script for replaying demonstrations

## Enhanced Checkpoint System

The new checkpoint system provides true "load checkpoint" functionality that preserves all episode data up to the target frame.

### Key Features

1. **Data Preservation**: When loading a checkpoint, all episode data up to the target frame is preserved
2. **Environment Reset**: Environment is reset to the exact state at the target frame
3. **Episode Truncation**: Episode data is automatically truncated to maintain consistency
4. **Error Handling**: Comprehensive error handling with user feedback

### Functions

#### `reset_and_keep_to(env, episode_data, target_frame_index)`
- Resets environment to specific frame while preserving episode data
- Returns `True` if successful, `False` otherwise

#### `save_checkpoint(env, checkpoint_path)`
- Saves current frame index to checkpoint file
- Returns `True` if successful, `False` otherwise

#### `load_checkpoint(env, checkpoint_path)`
- Loads checkpoint and resets to saved frame with data preservation
- Returns `True` if successful, `False` otherwise

#### `quick_rewind(env, frames_back=10)`
- Quickly goes back specified number of frames while preserving data
- Default: 10 frames back

### Usage

#### In Code
```python
from lw_benchhub.scripts.teleop.teleop_utils import (
    reset_and_keep_to,
    save_checkpoint,
    load_checkpoint,
    quick_rewind
)

# Save checkpoint
save_checkpoint(env, "checkpoint.pt")

# Load checkpoint
load_checkpoint(env, "checkpoint.pt")

# Reset to specific frame
reset_and_keep_to(env, episode_data, frame_index)

# Quick rewind
quick_rewind(env, frames_back=15)
```

#### Keyboard Shortcuts
- **M key**: Save checkpoint
- **N key**: Load checkpoint (with data preservation)
- **B key**: Quick rewind 10 frames (with data preservation)
- **R key**: Reset recording instance (clears all data)

### How It Works

1. **Backup**: Complete episode data is backed up before reset
2. **State Retrieval**: Target frame state is retrieved from episode data
3. **Environment Reset**: Environment is reset to target state using `env.reset_to()`
4. **Data Restoration**: Episode data is restored and truncated to target frame
5. **Consistency**: All data structures maintain consistency with the reset point

### Benefits

- **True Load Functionality**: Unlike standard `env.reset()`, this preserves all recorded data
- **Data Analysis**: Maintains complete episode history for analysis and replay
- **Error Recovery**: Allows users to go back to any previous point without losing data
- **Training Continuity**: Useful for RL training where episode data needs to be preserved

### Error Handling

The system provides comprehensive error handling:
- Invalid frame indices
- Missing episode data
- Environment reset failures
- Data restoration failures

All errors are logged with clear messages to help users understand what went wrong.

## Requirements

- Isaac Lab environment
- PyTorch
- Episode data with states recorded
- Recorder manager with active episodes

## Notes

- This system works best with environments that have active recorders
- Episode data must contain valid state information
- The system automatically handles device placement for tensor data
- All operations are logged for debugging purposes
