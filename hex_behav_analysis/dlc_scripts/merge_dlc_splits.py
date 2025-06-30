#!/usr/bin/env python3
"""
Merge split DeepLabCut analysis outputs back into single files.
This script combines CSV, H5, and pickle files from split video analysis.
Includes cleanup of partial files and robust error handling.
"""

import pandas as pd
import pickle
import h5py
import numpy as np
from pathlib import Path
import sys
import re
from collections import defaultdict
import shutil
import glob


def find_split_files(video_path, splits_per_video):
    """
    Find all split output files for a given video.
    
    Args:
        video_path: Path to the original video file
        splits_per_video: Number of splits that were created
        
    Returns:
        dict: Dictionary mapping split indices to file paths for each file type
    """
    video_path = Path(video_path)
    video_dir = video_path.parent
    video_stem = video_path.stem
    
    # Pattern to match split files
    # Format: videoname_frames{start}-{end}_split{index}of{total}*
    pattern = re.compile(
        rf"{re.escape(video_stem)}_frames(\d+)-(\d+)_split(\d+)of{splits_per_video}(.*)"
    )
    
    split_files = defaultdict(lambda: defaultdict(dict))
    
    # Search for all matching files
    for file in video_dir.iterdir():
        match = pattern.match(file.name)
        if match:
            start_frame = int(match.group(1))
            end_frame = int(match.group(2))
            split_index = int(match.group(3))
            suffix = match.group(4)
            
            # Determine file type from suffix
            if file.suffix == '.csv':
                file_type = 'csv'
            elif file.suffix == '.h5':
                file_type = 'h5'
            elif file.suffix == '.pickle':
                file_type = 'pickle'
            else:
                continue
            
            split_files[split_index][file_type] = {
                'path': file,
                'start_frame': start_frame,
                'end_frame': end_frame,
                'suffix': suffix
            }
    
    return split_files


def clean_old_merged_files(video_path, dlc_suffix):
    """
    Remove any existing merged files from previous runs.
    
    Args:
        video_path: Path to the original video
        dlc_suffix: The DLC model suffix
    """
    video_path = Path(video_path)
    video_dir = video_path.parent
    video_stem = video_path.stem
    
    # Patterns for merged files
    patterns = [
        f"{video_stem}{dlc_suffix}.csv",
        f"{video_stem}{dlc_suffix}.h5",
        f"{video_stem}{dlc_suffix}.pickle",
        f"{video_stem}{dlc_suffix}_meta.pickle"
    ]
    
    print("  Checking for existing merged files...")
    for pattern in patterns:
        for file_path in video_dir.glob(pattern):
            try:
                file_path.unlink()
                print(f"    Removed old merged file: {file_path.name}")
            except Exception as e:
                print(f"    Warning: Could not remove {file_path.name}: {e}")


def clean_partial_files(video_path):
    """
    Clean up any partial or temporary files from failed runs.
    
    Args:
        video_path: Path to the original video
    """
    video_path = Path(video_path)
    video_dir = video_path.parent
    video_stem = video_path.stem
    
    # Patterns for temporary/partial files
    patterns = [
        f".{video_stem}_split*_in_progress",  # Progress markers
        f"{video_stem}_split*_temp*",         # Temporary files
        f"{video_stem}_split*_partial*",      # Partial files
        f"{video_stem}_merged_temp*"          # Temporary merged files
    ]
    
    print("  Cleaning up partial files...")
    for pattern in patterns:
        for file_path in video_dir.glob(pattern):
            try:
                if file_path.is_file():
                    file_path.unlink()
                    print(f"    Removed partial file: {file_path.name}")
                elif file_path.is_dir():
                    shutil.rmtree(file_path)
                    print(f"    Removed partial directory: {file_path.name}")
            except Exception as e:
                print(f"    Warning: Could not remove {file_path.name}: {e}")


def merge_csv_files(split_files, output_path):
    """
    Merge CSV files from splits into a single CSV.
    
    Args:
        split_files: Dictionary of split file information
        output_path: Path for the merged CSV file
    """
    print(f"  Merging CSV files to: {output_path.name}")
    
    # Create temporary file for writing
    temp_path = output_path.parent / f"{output_path.stem}_merged_temp.csv"
    
    try:
        # Read all CSV files and store with their frame ranges
        dfs = []
        
        for split_index in sorted(split_files.keys()):
            if 'csv' in split_files[split_index]:
                csv_info = split_files[split_index]['csv']
                csv_path = csv_info['path']
                start_frame = csv_info['start_frame']
                
                print(f"    Reading split {split_index}: {csv_path.name}")
                
                # Read CSV with multi-level header
                df = pd.read_csv(csv_path, header=[0, 1, 2], index_col=0)
                
                # Adjust frame indices to match original video
                df.index = df.index + start_frame
                
                dfs.append(df)
        
        # Concatenate all dataframes
        if dfs:
            merged_df = pd.concat(dfs, axis=0)
            merged_df.sort_index(inplace=True)
            
            # Check for duplicate indices
            if merged_df.index.duplicated().any():
                print(f"    Warning: Found duplicate frame indices, keeping first occurrence")
                merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
            
            # Write to temporary file first
            merged_df.to_csv(temp_path)
            
            # Move to final location
            temp_path.rename(output_path)
            print(f"    Merged {len(dfs)} CSV files, total frames: {len(merged_df)}")
        else:
            print("    No CSV files found to merge")
            
    except Exception as e:
        # Clean up temporary file on error
        if temp_path.exists():
            temp_path.unlink()
        raise e


def merge_h5_files(split_files, output_path):
    """
    Merge H5 files from splits into a single H5 file.
    
    Args:
        split_files: Dictionary of split file information
        output_path: Path for the merged H5 file
    """
    print(f"  Merging H5 files to: {output_path.name}")
    
    # Create temporary file for writing
    temp_path = output_path.parent / f"{output_path.stem}_merged_temp.h5"
    
    try:
        # Collect all data from split files
        all_data = []
        scorer = None
        bodyparts = None
        coords = None
        
        for split_index in sorted(split_files.keys()):
            if 'h5' in split_files[split_index]:
                h5_info = split_files[split_index]['h5']
                h5_path = h5_info['path']
                start_frame = h5_info['start_frame']
                
                print(f"    Reading split {split_index}: {h5_path.name}")
                
                with pd.HDFStore(h5_path, 'r') as store:
                    # Read the dataframe
                    df_key = [key for key in store.keys() if 'df_with_missing' in key][0]
                    df = store[df_key]
                    
                    # Get metadata from first file
                    if scorer is None:
                        scorer = df.columns.levels[0][0]
                        bodyparts = df.columns.levels[1].tolist()
                        coords = df.columns.levels[2].tolist()
                    
                    # Adjust frame indices
                    df.index = df.index + start_frame
                    all_data.append(df)
        
        # Merge all data
        if all_data:
            merged_df = pd.concat(all_data, axis=0)
            merged_df.sort_index(inplace=True)
            
            # Check for duplicate indices
            if merged_df.index.duplicated().any():
                print(f"    Warning: Found duplicate frame indices, keeping first occurrence")
                merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
            
            # Write to temporary file first
            with pd.HDFStore(temp_path, 'w') as store:
                store['df_with_missing'] = merged_df
                store.get_storer('df_with_missing').attrs['scorer'] = scorer
                store.get_storer('df_with_missing').attrs['bodyparts'] = bodyparts
                store.get_storer('df_with_missing').attrs['coords'] = coords
            
            # Move to final location
            temp_path.rename(output_path)
            print(f"    Merged {len(all_data)} H5 files, total frames: {len(merged_df)}")
        else:
            print("    No H5 files found to merge")
            
    except Exception as e:
        # Clean up temporary file on error
        if temp_path.exists():
            temp_path.unlink()
        raise e


def merge_pickle_files(split_files, output_path):
    """
    Merge pickle metadata files from splits.
    For metadata, we typically just need one file as they should be identical.
    
    Args:
        split_files: Dictionary of split file information
        output_path: Path for the merged pickle file
    """
    print(f"  Copying pickle metadata to: {output_path.name}")
    
    # Create temporary file for writing
    temp_path = output_path.parent / f"{output_path.stem}_merged_temp.pickle"
    
    try:
        # Find the first pickle file
        for split_index in sorted(split_files.keys()):
            if 'pickle' in split_files[split_index]:
                pickle_path = split_files[split_index]['pickle']['path']
                
                print(f"    Using metadata from split {split_index}: {pickle_path.name}")
                
                # Read and write the pickle file
                with open(pickle_path, 'rb') as f:
                    metadata = pickle.load(f)
                
                with open(temp_path, 'wb') as f:
                    pickle.dump(metadata, f)
                
                # Move to final location
                temp_path.rename(output_path)
                print("    Metadata copied successfully")
                break
                
    except Exception as e:
        # Clean up temporary file on error
        if temp_path.exists():
            temp_path.unlink()
        raise e


def clean_split_files(split_files):
    """
    Remove temporary split files after successful merge.
    
    Args:
        split_files: Dictionary of split file information
    """
    print("  Cleaning up split files...")
    
    for split_index, file_types in split_files.items():
        for file_type, file_info in file_types.items():
            file_path = file_info['path']
            if file_path.exists():
                file_path.unlink()
                print(f"    Removed: {file_path.name}")


def verify_merge_completeness(video_path, splits_per_video, split_files):
    """
    Verify that we have a complete set of splits before merging.
    Only proceeds if ALL splits are present.
    
    Args:
        video_path: Path to the original video
        splits_per_video: Expected number of splits
        split_files: Found split files
        
    Returns:
        bool: True if merge should proceed (all splits present), False otherwise
    """
    expected_splits = set(range(1, splits_per_video + 1))
    found_splits = set(split_files.keys())
    
    if found_splits == expected_splits:
        print(f"  ✓ All {splits_per_video} splits found")
        return True
    
    missing = expected_splits - found_splits
    print(f"  ✗ Missing splits: {sorted(missing)}")
    print(f"  ! Skipping merge - requires all {splits_per_video} splits to be present")
    return False

def process_video(video_path, splits_per_video, cleanup=True, force_merge=False):
    """
    Process all splits for a single video and merge them.
    
    Args:
        video_path: Path to the original video file
        splits_per_video: Number of splits that were created
        cleanup: Whether to remove split files after merging
        force_merge: Whether to force merge even with missing splits
    """
    video_path = Path(video_path)
    print(f"\nProcessing video: {video_path.name}")
    
    # Clean partial files first
    clean_partial_files(video_path)
    
    # Find all split files
    split_files = find_split_files(video_path, splits_per_video)
    
    if not split_files:
        print(f"  WARNING: No split files found for {video_path.name}")
        return
    
    print(f"  Found splits: {sorted(split_files.keys())}")
    
    # Verify completeness
    if not force_merge and not verify_merge_completeness(video_path, splits_per_video, split_files):
        print(f"  Skipping merge for {video_path.name} due to missing splits")
        return
    
    # Get the DLC suffix from one of the CSV files specifically
    # This avoids picking up _meta.pickle suffixes
    dlc_suffix = None
    for split_index in sorted(split_files.keys()):
        if 'csv' in split_files[split_index]:
            suffix = split_files[split_index]['csv']['suffix']
            # Remove the .csv extension from the suffix if present
            dlc_suffix = suffix.replace('.csv', '')
            break
    
    if not dlc_suffix:
        print("  ERROR: Could not determine DLC suffix from CSV files")
        # Still try to clean up split files before returning
        if cleanup:
            clean_split_files(split_files)
        return
    
    # Clean old merged files
    clean_old_merged_files(video_path, dlc_suffix)
    
    # Define output paths
    output_base = video_path.parent / f"{video_path.stem}{dlc_suffix}"
    
    # Create progress marker
    merge_marker = video_path.parent / f".{video_path.stem}_merge_in_progress"
    merge_marker.touch()
    
    try:
        # Merge CSV files
        if any('csv' in split_files[i] for i in split_files):
            merge_csv_files(split_files, Path(str(output_base) + '.csv'))
        
        # Merge H5 files (only if pytables is available)
        if any('h5' in split_files[i] for i in split_files):
            try:
                merge_h5_files(split_files, Path(str(output_base) + '.h5'))
            except ImportError as e:
                print(f"  Warning: Could not merge H5 files - {str(e)}")
                print("  Continuing with CSV merge only...")
        
        # Copy pickle file (only if we care about it)
        if any('pickle' in split_files[i] for i in split_files):
            try:
                merge_pickle_files(split_files, Path(str(output_base) + '.pickle'))
            except Exception as e:
                print(f"  Warning: Could not copy pickle file - {str(e)}")
        
        print(f"  ✓ Merge complete for {video_path.name}")
        
    except Exception as e:
        print(f"  ✗ ERROR during merge: {str(e)}")
        # Don't re-raise since we want to clean up regardless
    finally:
        # Always clean up split files if requested, regardless of merge success
        if cleanup:
            print("  Cleaning up split files (regardless of merge status)...")
            clean_split_files(split_files)
        
        # Remove progress marker
        if merge_marker.exists():
            merge_marker.unlink()


def clean_temp_directories(video_paths):
    """
    Clean up temporary directories after processing.
    
    Args:
        video_paths: List of video paths that were processed
    """
    print("\nCleaning up temporary directories...")
    
    for video_path in video_paths:
        video_dir = Path(video_path).parent
        
        # Clean up temp_splits directory
        temp_dir = video_dir / "temp_splits"
        if temp_dir.exists():
            try:
                # Remove any remaining files
                for file in temp_dir.iterdir():
                    file.unlink()
                    print(f"  Removed leftover file: {file.name}")
                
                # Remove directory if empty
                if not any(temp_dir.iterdir()):
                    temp_dir.rmdir()
                    print(f"  Removed temp directory: {temp_dir}")
            except Exception as e:
                print(f"  Warning: Could not clean {temp_dir}: {e}")


def main():
    """Main function to process all videos in the list."""
    if len(sys.argv) != 3:
        print("Usage: python merge_dlc_splits.py VIDEO_LIST_FILE SPLITS_PER_VIDEO")
        sys.exit(1)
    
    video_list_file = Path(sys.argv[1])
    splits_per_video = int(sys.argv[2])
    
    print(f"=== DeepLabCut Split Merge Script ===")
    print(f"Video list: {video_list_file}")
    print(f"Splits per video: {splits_per_video}")
    
    # Check if video list exists
    if not video_list_file.exists():
        print(f"ERROR: Video list file not found: {video_list_file}")
        sys.exit(1)
    
    # Read video list
    with open(video_list_file, 'r') as f:
        video_paths = [line.strip() for line in f if line.strip()]
    
    print(f"Found {len(video_paths)} videos to process")
    
    # Process each video
    success_count = 0
    failed_videos = []
    
    for video_path in video_paths:
        try:
            process_video(video_path, splits_per_video, cleanup=True, force_merge=False)
            success_count += 1
        except Exception as e:
            print(f"\nERROR processing {video_path}: {str(e)}")
            failed_videos.append(video_path)
            # Don't print full traceback to keep output cleaner
    
    # Final cleanup
    clean_temp_directories(video_paths)
    
    print(f"\n=== Merge Summary ===")
    print(f"Successfully processed: {success_count}/{len(video_paths)} videos")
    
    if failed_videos:
        print(f"\nFailed videos ({len(failed_videos)}):")
        for video in failed_videos:
            print(f"  - {video}")


if __name__ == "__main__":
    main()