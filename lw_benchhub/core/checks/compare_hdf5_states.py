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

"""
HDF5 State Comparison Script

This script compares state values between two HDF5 files and reports differences
in state categories (articulation, rigid_object, deformable_object, etc.).

Usage:
    python compare_hdf5_states.py <file1.hdf5> <file2.hdf5> [--tolerance 1e-6] [--verbose]
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import h5py
import numpy as np


class HDF5StateComparator:
    """Compare state values between two HDF5 files."""

    def __init__(self, file1_path: str, file2_path: str, tolerance: float = 1e-6):
        self.file1_path = file1_path
        self.file2_path = file2_path
        self.tolerance = tolerance
        self.differences = []

    def compare_files(self) -> Dict[str, Any]:
        """Compare all state data between the two HDF5 files."""
        print(f"Comparing HDF5 files:")
        print(f"  File 1: {self.file1_path}")
        print(f"  File 2: {self.file2_path}")
        print(f"  Tolerance: {self.tolerance}")
        print()

        with h5py.File(self.file1_path, 'r') as f1, h5py.File(self.file2_path, 'r') as f2:
            # Compare file structure
            structure_diff = self._compare_structure(f1, f2)

            # Compare state data
            state_differences = self._compare_state_data(f1, f2)

            # Generate summary
            summary = self._generate_summary(structure_diff, state_differences)

            return {
                'structure_differences': structure_diff,
                'state_differences': state_differences,
                'summary': summary
            }

    def _compare_structure(self, f1: h5py.File, f2: h5py.File) -> Dict[str, Any]:
        """Compare the overall structure of the two HDF5 files."""
        print("Comparing file structure...")

        def get_all_keys(group, prefix=""):
            """Recursively get all keys in an HDF5 group."""
            keys = set()
            for key in group.keys():
                full_key = f"{prefix}/{key}" if prefix else key
                keys.add(full_key)
                if isinstance(group[key], h5py.Group):
                    keys.update(get_all_keys(group[key], full_key))
            return keys

        keys1 = get_all_keys(f1)
        keys2 = get_all_keys(f2)

        only_in_file1 = keys1 - keys2
        only_in_file2 = keys2 - keys1
        common_keys = keys1 & keys2

        structure_diff = {
            'only_in_file1': list(only_in_file1),
            'only_in_file2': list(only_in_file2),
            'common_keys': list(common_keys),
            'total_keys_file1': len(keys1),
            'total_keys_file2': len(keys2)
        }

        if only_in_file1:
            print(f"  Keys only in file 1: {len(only_in_file1)}")
        if only_in_file2:
            print(f"  Keys only in file 2: {len(only_in_file2)}")
        print(f"  Common keys: {len(common_keys)}")

        return structure_diff

    def _compare_state_data(self, f1: h5py.File, f2: h5py.File) -> Dict[str, Any]:
        """Compare state data between the two files."""
        print("\nComparing state data...")

        state_differences = {
            'episode_differences': {},
            'category_differences': {},
            'total_episodes_file1': 0,
            'total_episodes_file2': 0
        }

        # Get all episodes from both files
        episodes1 = self._get_episodes(f1)
        episodes2 = self._get_episodes(f2)

        state_differences['total_episodes_file1'] = len(episodes1)
        state_differences['total_episodes_file2'] = len(episodes2)

        print(f"  Episodes in file 1: {len(episodes1)}")
        print(f"  Episodes in file 2: {len(episodes2)}")

        # Compare common episodes
        common_episodes = set(episodes1) & set(episodes2)
        only_in_file1_episodes = set(episodes1) - set(episodes2)
        only_in_file2_episodes = set(episodes2) - set(episodes1)

        if only_in_file1_episodes:
            print(f"  Episodes only in file 1: {len(only_in_file1_episodes)}")
        if only_in_file2_episodes:
            print(f"  Episodes only in file 2: {len(only_in_file2_episodes)}")
        print(f"  Common episodes: {len(common_episodes)}")

        # Compare state data for common episodes
        for episode in common_episodes:
            print(f"\n  Comparing episode: {episode}")
            episode_diff = self._compare_episode_states(f1, f2, episode)
            if episode_diff['has_differences']:
                state_differences['episode_differences'][episode] = episode_diff

        # Update category differences summary
        self._update_category_summary(state_differences)

        return state_differences

    def _get_episodes(self, f: h5py.File) -> List[str]:
        """Get all episode names from an HDF5 file."""
        if 'data' not in f:
            return []
        return [key for key in f['data'].keys() if key.startswith('demo_')]

    def _compare_episode_states(self, f1: h5py.File, f2: h5py.File, episode: str) -> Dict[str, Any]:
        """Compare state data for a specific episode."""
        episode_diff = {
            'has_differences': False,
            'category_differences': {},
            'missing_categories': {'file1': [], 'file2': []},
            'missing_entities': {'file1': {}, 'file2': {}}
        }

        # Check if states exist in both files
        states_path1 = f"data/{episode}/states"
        states_path2 = f"data/{episode}/states"

        if states_path1 not in f1:
            episode_diff['missing_categories']['file1'].append('states')
            return episode_diff
        if states_path2 not in f2:
            episode_diff['missing_categories']['file2'].append('states')
            return episode_diff

        states1 = f1[states_path1]
        states2 = f2[states_path2]

        # Compare each state category (articulation, rigid_object, deformable_object)
        for category in ['articulation', 'rigid_object', 'deformable_object']:
            if category in states1 and category in states2:
                category_diff = self._compare_category(states1[category], states2[category], category)
                if category_diff['has_differences']:
                    episode_diff['category_differences'][category] = category_diff
                    episode_diff['has_differences'] = True
            elif category in states1:
                episode_diff['missing_categories']['file2'].append(f'states/{category}')
                episode_diff['has_differences'] = True
            elif category in states2:
                episode_diff['missing_categories']['file1'].append(f'states/{category}')
                episode_diff['has_differences'] = True

        return episode_diff

    def _compare_category(self, cat1: h5py.Group, cat2: h5py.Group, category_name: str) -> Dict[str, Any]:
        """Compare a specific state category between two files."""
        category_diff = {
            'has_differences': False,
            'entity_differences': {},
            'missing_entities': {'file1': [], 'file2': []}
        }

        entities1 = set(cat1.keys())
        entities2 = set(cat2.keys())

        # Check for missing entities
        missing_in_file2 = entities1 - entities2
        missing_in_file1 = entities2 - entities1

        if missing_in_file2:
            category_diff['missing_entities']['file2'] = list(missing_in_file2)
            category_diff['has_differences'] = True
        if missing_in_file1:
            category_diff['missing_entities']['file1'] = list(missing_in_file1)
            category_diff['has_differences'] = True

        # Compare common entities
        common_entities = entities1 & entities2
        for entity in common_entities:
            entity_diff = self._compare_entity(cat1[entity], cat2[entity], entity, category_name)
            if entity_diff['has_differences']:
                category_diff['entity_differences'][entity] = entity_diff
                category_diff['has_differences'] = True

        return category_diff

    def _compare_entity(self, entity1: h5py.Group, entity2: h5py.Group, entity_name: str, category_name: str) -> Dict[str, Any]:
        """Compare a specific entity between two files."""
        entity_diff = {
            'has_differences': False,
            'state_differences': {},
            'missing_states': {'file1': [], 'file2': []}
        }

        states1 = set(entity1.keys())
        states2 = set(entity2.keys())

        # Check for missing states
        missing_in_file2 = states1 - states2
        missing_in_file1 = states2 - states1

        if missing_in_file2:
            entity_diff['missing_states']['file2'] = list(missing_in_file2)
            entity_diff['has_differences'] = True
        if missing_in_file1:
            entity_diff['missing_states']['file1'] = list(missing_in_file1)
            entity_diff['has_differences'] = True

        # Compare common states
        common_states = states1 & states2
        for state in common_states:
            state_diff = self._compare_state_values(entity1[state], entity2[state], state, entity_name, category_name)
            if state_diff['has_differences']:
                entity_diff['state_differences'][state] = state_diff
                entity_diff['has_differences'] = True

        return entity_diff

    def _compare_state_values(self, state1: h5py.Dataset, state2: h5py.Dataset, state_name: str, entity_name: str, category_name: str) -> Dict[str, Any]:
        """Compare values of a specific state between two files."""
        state_diff = {
            'has_differences': False,
            'shape_difference': False,
            'value_differences': {},
            'max_difference': 0.0,
            'mean_difference': 0.0,
            'shape1': state1.shape,
            'shape2': state2.shape,
            'dtype1': str(state1.dtype),
            'dtype2': str(state2.dtype)
        }

        # Check shape differences
        if state1.shape != state2.shape:
            state_diff['shape_difference'] = True
            state_diff['has_differences'] = True
            return state_diff

        # Check dtype differences
        if state1.dtype != state2.dtype:
            state_diff['has_differences'] = True
            return state_diff

        # Compare values
        try:
            data1 = state1[:]
            data2 = state2[:]

            if np.array_equal(data1, data2):
                return state_diff

            # Calculate differences
            if np.issubdtype(data1.dtype, np.floating):
                diff = np.abs(data1 - data2)
                max_diff = np.max(diff)
                mean_diff = np.mean(diff)

                state_diff['max_difference'] = float(max_diff)
                state_diff['mean_difference'] = float(mean_diff)

                if max_diff > self.tolerance:
                    state_diff['has_differences'] = True

                    # Find locations with significant differences
                    significant_diff_mask = diff > self.tolerance
                    if np.any(significant_diff_mask):
                        diff_indices = np.where(significant_diff_mask)
                        state_diff['value_differences'] = {
                            'indices_with_differences': [idx.tolist() for idx in diff_indices],
                            'max_diff_value': float(np.max(diff[significant_diff_mask])),
                            'num_different_elements': int(np.sum(significant_diff_mask))
                        }
            else:
                # For non-numeric data, check for exact equality
                if not np.array_equal(data1, data2):
                    state_diff['has_differences'] = True
                    state_diff['value_differences'] = {
                        'type': 'non_numeric_difference',
                        'description': 'Non-numeric data differs between files'
                    }

        except Exception as e:
            state_diff['has_differences'] = True
            state_diff['value_differences'] = {
                'error': str(e),
                'type': 'comparison_error'
            }

        return state_diff

    def _update_category_summary(self, state_differences: Dict[str, Any]):
        """Update category-level summary of differences."""
        category_summary = {}

        for episode, episode_diff in state_differences['episode_differences'].items():
            for category, category_diff in episode_diff['category_differences'].items():
                if category not in category_summary:
                    category_summary[category] = {
                        'episodes_with_differences': 0,
                        'total_entities_different': 0,
                        'total_states_different': 0,
                        'max_difference': 0.0,
                        'mean_difference': 0.0
                    }

                category_summary[category]['episodes_with_differences'] += 1

                for entity, entity_diff in category_diff['entity_differences'].items():
                    category_summary[category]['total_entities_different'] += 1

                    for state, state_diff in entity_diff['state_differences'].items():
                        category_summary[category]['total_states_different'] += 1
                        category_summary[category]['max_difference'] = max(
                            category_summary[category]['max_difference'],
                            state_diff['max_difference']
                        )
                        category_summary[category]['mean_difference'] = max(
                            category_summary[category]['mean_difference'],
                            state_diff['mean_difference']
                        )

        state_differences['category_differences'] = category_summary

    def _generate_summary(self, structure_diff: Dict[str, Any], state_differences: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of all differences found."""
        summary = {
            'files_identical': False,
            'total_episodes_different': len(state_differences['episode_differences']),
            'categories_with_differences': list(state_differences['category_differences'].keys()),
            'structure_differences': {
                'keys_only_in_file1': len(structure_diff['only_in_file1']),
                'keys_only_in_file2': len(structure_diff['only_in_file2']),
                'common_keys': len(structure_diff['common_keys'])
            }
        }

        # Check if files are identical
        summary['files_identical'] = (
            len(structure_diff['only_in_file1']) == 0 and
            len(structure_diff['only_in_file2']) == 0 and
            len(state_differences['episode_differences']) == 0
        )

        return summary

    def print_detailed_report(self, comparison_result: Dict[str, Any], verbose: bool = False):
        """Print a detailed report of the comparison results."""
        print("\n" + "=" * 80)
        print("DETAILED COMPARISON REPORT")
        print("=" * 80)

        summary = comparison_result['summary']
        structure_diff = comparison_result['structure_differences']
        state_diff = comparison_result['state_differences']

        # Overall summary
        print(f"\nOVERALL SUMMARY:")
        print(f"  Files identical: {summary['files_identical']}")
        print(f"  Episodes with differences: {summary['total_episodes_different']}")
        print(f"  Categories with differences: {len(summary['categories_with_differences'])}")

        # Structure differences
        if structure_diff['only_in_file1'] or structure_diff['only_in_file2']:
            print(f"\nSTRUCTURE DIFFERENCES:")
            print(f"  Keys only in file 1: {len(structure_diff['only_in_file1'])}")
            print(f"  Keys only in file 2: {len(structure_diff['only_in_file2'])}")
            if verbose:
                if structure_diff['only_in_file1']:
                    print(f"    File 1 only: {structure_diff['only_in_file1'][:10]}...")
                if structure_diff['only_in_file2']:
                    print(f"    File 2 only: {structure_diff['only_in_file2'][:10]}...")

        # State differences by category
        if summary['categories_with_differences']:
            print(f"\nSTATE DIFFERENCES BY CATEGORY:")
            for category, cat_info in state_diff['category_differences'].items():
                print(f"\n  {category.upper()}:")
                print(f"    Episodes with differences: {cat_info['episodes_with_differences']}")
                print(f"    Total entities different: {cat_info['total_entities_different']}")
                print(f"    Total states different: {cat_info['total_states_different']}")
                print(f"    Max difference: {cat_info['max_difference']:.2e}")
                print(f"    Mean difference: {cat_info['mean_difference']:.2e}")

        # Episode-level details (if verbose)
        if verbose and state_diff['episode_differences']:
            print(f"\nEPISODE-LEVEL DETAILS:")
            for episode, episode_info in state_diff['episode_differences'].items():
                print(f"\n  Episode {episode}:")
                for category, category_info in episode_info['category_differences'].items():
                    print(f"    {category}:")
                    for entity, entity_info in category_info['entity_differences'].items():
                        print(f"      {entity}:")
                        for state, state_info in entity_info['state_differences'].items():
                            print(f"        {state}: max_diff={state_info['max_difference']:.2e}, mean_diff={state_info['mean_difference']:.2e}")


def main():
    parser = argparse.ArgumentParser(description='Compare state values between two HDF5 files')
    parser.add_argument('file1', help='First HDF5 file path')
    parser.add_argument('file2', help='Second HDF5 file path')
    parser.add_argument('--tolerance', type=float, default=1e-6, help='Tolerance for numerical comparisons')
    parser.add_argument('--verbose', action='store_true', help='Print detailed information')
    parser.add_argument('--output', help='Output file for detailed report (JSON format)')

    args = parser.parse_args()

    # Validate input files
    if not Path(args.file1).exists():
        print(f"Error: File {args.file1} does not exist")
        return 1

    if not Path(args.file2).exists():
        print(f"Error: File {args.file2} does not exist")
        return 1

    # Perform comparison
    try:
        comparator = HDF5StateComparator(args.file1, args.file2, args.tolerance)
        result = comparator.compare_files()

        # Print report
        comparator.print_detailed_report(result, args.verbose)

        # Save detailed report if requested
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nDetailed report saved to: {args.output}")

        # Return appropriate exit code
        return 0 if result['summary']['files_identical'] else 1

    except Exception as e:
        print(f"Error during comparison: {e}")
        return 1


if __name__ == '__main__':
    exit(main())
