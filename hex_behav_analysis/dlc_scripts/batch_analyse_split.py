#!/usr/bin/env python3
"""
Split video analysis script for DeepLabCut.
Analyses a specific portion of a video to enable parallel processing.
"""
super_animal = False  # Set to True to use SuperAnimal, False for standard DeepLabCut

if super_animal:
    from deeplabcut import video_inference_superanimal
else:
    from deeplabcut import analyze_videos

import time
import sys
import cv2
import os
from pathlib import Path
import shutil
import glob

# Configuration path
if not super_animal:
    config = r'/cephfs2/srogers/DEEPLABCUT_models/LMDC_model_videos/models/LMDC-StefanRC-2025-03-11/config.yaml'
    # config = r'/cephfs2/srogers/DEEPLABCUT_models/2500601_Pitx2_ephys_model/project_folders/tetrodes-StefanRC-2025-06-01/config.yaml'

if super_animal:
    def get_video_info(video_path):
        """
        Get total frame count and FPS from video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            tuple: (total_frames, fps)
        """
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return total_frames, fps


    def calculate_split_frames(total_frames, split_index, total_splits):
        """
        Calculate start and end frames for a specific split.
        
        Args:
            total_frames: Total number of frames in the video
            split_index: Current split number (1-based)
            total_splits: Total number of splits
            
        Returns:
            tuple: (start_frame, end_frame)
        """
        frames_per_split = total_frames // total_splits
        remainder = total_frames % total_splits
        
        # Calculate start frame
        start_frame = (split_index - 1) * frames_per_split
        
        # Add remainder frames to earlier splits
        if split_index <= remainder:
            start_frame += (split_index - 1)
            frames_per_split += 1
        else:
            start_frame += remainder
        
        # Calculate end frame
        end_frame = start_frame + frames_per_split - 1
        
        # Ensure we don't exceed total frames
        end_frame = min(end_frame, total_frames - 1)
        
        return start_frame, end_frame


    def create_split_video_ffmpeg(video_path, output_path, start_frame, end_frame, fps):
        """
        Create a temporary video file containing only the specified frame range using ffmpeg.
        This is much faster than using OpenCV as it can seek and copy without re-encoding.
        
        Args:
            video_path: Path to the original video
            output_path: Path for the temporary split video
            start_frame: First frame to include
            end_frame: Last frame to include
            fps: Frames per second of the video
            
        Returns:
            bool: True if successful, False otherwise
        """
        import subprocess
        
        # Calculate start time and duration in seconds
        start_time = start_frame / fps
        duration = (end_frame - start_frame + 1) / fps
        
        # Build ffmpeg command
        # -ss: seek to start time (before input for fast seeking)
        # -i: input file
        # -t: duration to extract
        # -c:v copy: copy video codec (no re-encoding)
        # -avoid_negative_ts make_zero: fix timestamp issues
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),  # Seek to start time
            '-i', str(video_path),   # Input file
            '-t', str(duration),     # Duration to extract
            '-c:v', 'copy',          # Copy video stream without re-encoding
            '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
            '-y',                    # Overwrite output file if exists
            str(output_path)
        ]
        
        print(f"Running ffmpeg command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("FFmpeg split completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr}")
            return False


    def create_split_video_opencv(video_path, output_path, start_frame, end_frame):
        """
        Create a temporary video file containing only the specified frame range using OpenCV.
        This is the fallback method if ffmpeg is not available.
        
        Args:
            video_path: Path to the original video
            output_path: Path for the temporary split video
            start_frame: First frame to include
            end_frame: Last frame to include
            
        Returns:
            bool: True if successful, False otherwise
        """
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Define the codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Set position to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Write frames from start to end
        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_idx}")
                break
            out.write(frame)
        
        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        return True


    def create_split_video(video_path, output_path, start_frame, end_frame, fps):
        """
        Create a temporary video file containing only the specified frame range.
        Tries ffmpeg first for speed, falls back to OpenCV if needed.
        
        Args:
            video_path: Path to the original video
            output_path: Path for the temporary split video
            start_frame: First frame to include
            end_frame: Last frame to include
            fps: Frames per second of the video
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if ffmpeg is available
        try:
            import subprocess
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            use_ffmpeg = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("FFmpeg not found, falling back to OpenCV method")
            use_ffmpeg = False
        
        if use_ffmpeg:
            return create_split_video_ffmpeg(video_path, output_path, start_frame, end_frame, fps)
        else:
            return create_split_video_opencv(video_path, output_path, start_frame, end_frame)


    def clean_partial_outputs(video_path, split_index, total_splits, start_frame, end_frame):
        """
        Clean up any partial output files from previous failed runs.
        
        Args:
            video_path: Path to the original video
            split_index: Current split number
            total_splits: Total number of splits
            start_frame: Start frame for this split
            end_frame: End frame for this split
        """
        video_dir = video_path.parent
        video_stem = video_path.stem
        
        # Patterns for files that might exist from previous runs
        patterns = [
            # Split-specific patterns
            f"{video_stem}_split{split_index}of{total_splits}*",
            f"{video_stem}_frames{start_frame}-{end_frame}_split{split_index}of{total_splits}*",
            # Temporary/partial files
            f"{video_stem}*_split{split_index}_temp*",
            f"{video_stem}*_split{split_index}_partial*",
        ]
        
        print("Cleaning up potential partial files from previous runs...")
        for pattern in patterns:
            for file_path in video_dir.glob(pattern):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                        print(f"  Removed partial file: {file_path.name}")
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
                        print(f"  Removed partial directory: {file_path.name}")
                except Exception as e:
                    print(f"  Warning: Could not remove {file_path.name}: {e}")


    def analyse_split(video_path, gpu_id, split_index, total_splits):
        """
        Analyse a specific split of a video using SuperAnimal.
        
        Args:
            video_path: Path to the video file
            gpu_id: GPU index to use
            split_index: Current split number (1-based)
            total_splits: Total number of splits
        """
        start_time = time.perf_counter()
        video_path = Path(video_path)
        
        print(f"Analysing split {split_index}/{total_splits} of {video_path.name} with SuperAnimal")
        print(f"Using GPU {gpu_id}")
        
        # Get video information
        total_frames, fps = get_video_info(video_path)
        print(f"Video info: {total_frames} frames at {fps:.2f} FPS")
        
        # Calculate split boundaries
        start_frame, end_frame = calculate_split_frames(total_frames, split_index, total_splits)
        print(f"Processing frames {start_frame} to {end_frame} ({end_frame - start_frame + 1} frames)")
        
        # Clean up any partial files from previous runs
        clean_partial_outputs(video_path, split_index, total_splits, start_frame, end_frame)
        
        # Create temporary split video
        temp_dir = video_path.parent / "temp_splits"
        temp_dir.mkdir(exist_ok=True)
        
        temp_video_name = f"{video_path.stem}_split{split_index}of{total_splits}.avi"
        temp_video_path = temp_dir / temp_video_name
        
        # Remove temp video if it already exists
        if temp_video_path.exists():
            print(f"Removing existing temp video: {temp_video_path}")
            temp_video_path.unlink()
        
        print(f"Creating temporary split video: {temp_video_path}")
        if not create_split_video(video_path, temp_video_path, start_frame, end_frame, fps):
            raise RuntimeError("Failed to create split video")
        
        # Analyse the split video with SuperAnimal
        try:
            # Create a marker file to indicate analysis is in progress
            progress_marker = video_path.parent / f".{video_path.stem}_split{split_index}_in_progress"
            progress_marker.touch()
            
            # Use SuperAnimal for analysis - no config needed!
            video_inference_superanimal(
                videos=[str(temp_video_path)],
                superanimal_name="superanimal_topviewmouse",
                model_name="hrnet_w32",
                detector_name="fasterrcnn_resnet50_fpn_v2",
                video_adapt=True,  # Helps reduce jitter
                videotype='.avi'
                # Note: SuperAnimal saves outputs in the same directory as the video by default
            )
            
            # Remove progress marker
            if progress_marker.exists():
                progress_marker.unlink()
            
            # Rename output files to include split information and frame range
            # SuperAnimal creates files in the same directory as the temp video
            base_pattern = temp_video_path.stem
            renamed_files = []
            
            # Look for SuperAnimal output files in the temp directory
            for file in temp_dir.glob(f"{base_pattern}*"):
                if file.suffix in ['.csv', '.h5', '.pickle']:
                    # Extract the SuperAnimal suffix from the filename
                    superanimal_suffix = file.name[len(base_pattern):]
                    
                    # Create new filename with frame range information
                    new_name = f"{video_path.stem}_frames{start_frame}-{end_frame}_split{split_index}of{total_splits}{superanimal_suffix}"
                    new_path = video_path.parent / new_name
                    
                    # Remove target file if it already exists (from previous run)
                    if new_path.exists():
                        print(f"  Removing existing output file: {new_path.name}")
                        new_path.unlink()
                    
                    # Move the file from temp directory to video directory
                    file.rename(new_path)
                    renamed_files.append(new_name)
                    print(f"  Moved and renamed output: {file.name} -> {new_name}")
            
            if not renamed_files:
                # Also check the video directory in case SuperAnimal put files there
                for file in video_path.parent.glob(f"{base_pattern}*"):
                    if file.suffix in ['.csv', '.h5', '.pickle']:
                        superanimal_suffix = file.name[len(base_pattern):]
                        new_name = f"{video_path.stem}_frames{start_frame}-{end_frame}_split{split_index}of{total_splits}{superanimal_suffix}"
                        new_path = file.parent / new_name
                        
                        if new_path.exists():
                            print(f"  Removing existing output file: {new_path.name}")
                            new_path.unlink()
                        
                        file.rename(new_path)
                        renamed_files.append(new_name)
                        print(f"  Renamed output: {file.name} -> {new_name}")
            
            if not renamed_files:
                raise RuntimeError("No output files were generated by SuperAnimal")
        
        except Exception as e:
            # Clean up on error
            if 'progress_marker' in locals() and progress_marker.exists():
                progress_marker.unlink()
            
            # Clean up any partial output files
            print("Cleaning up partial outputs due to error...")
            for pattern in [f"{temp_video_path.stem}*", f"{video_path.stem}_split{split_index}*"]:
                for file in video_path.parent.glob(pattern):
                    if file != temp_video_path:  # Don't remove the temp video yet
                        try:
                            file.unlink()
                            print(f"  Removed partial output: {file.name}")
                        except Exception as cleanup_error:
                            print(f"  Warning: Could not remove {file.name}: {cleanup_error}")
            
            raise e
        
        finally:
            # Always clean up temporary video
            if temp_video_path.exists():
                temp_video_path.unlink()
                print(f"Removed temporary video: {temp_video_path}")
            
            # Clean up temp directory if empty
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()
                print(f"Removed empty temp directory: {temp_dir}")
        
        # Calculate and print timing
        elapsed_time = time.perf_counter() - start_time
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        
        print(f"""SuperAnimal analysis of split {split_index}/{total_splits} complete. 
                    \rTook: {minutes} minutes and {seconds:.2f} seconds""")

else:
    def get_video_info(video_path):
        """
        Get total frame count and FPS from video.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            tuple: (total_frames, fps)
        """
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return total_frames, fps


    def calculate_split_frames(total_frames, split_index, total_splits):
        """
        Calculate start and end frames for a specific split.
        
        Args:
            total_frames: Total number of frames in the video
            split_index: Current split number (1-based)
            total_splits: Total number of splits
            
        Returns:
            tuple: (start_frame, end_frame)
        """
        frames_per_split = total_frames // total_splits
        remainder = total_frames % total_splits
        
        # Calculate start frame
        start_frame = (split_index - 1) * frames_per_split
        
        # Add remainder frames to earlier splits
        if split_index <= remainder:
            start_frame += (split_index - 1)
            frames_per_split += 1
        else:
            start_frame += remainder
        
        # Calculate end frame
        end_frame = start_frame + frames_per_split - 1
        
        # Ensure we don't exceed total frames
        end_frame = min(end_frame, total_frames - 1)
        
        return start_frame, end_frame


    def create_split_video_ffmpeg(video_path, output_path, start_frame, end_frame, fps):
        """
        Create a temporary video file containing only the specified frame range using ffmpeg.
        This is much faster than using OpenCV as it can seek and copy without re-encoding.
        
        Args:
            video_path: Path to the original video
            output_path: Path for the temporary split video
            start_frame: First frame to include
            end_frame: Last frame to include
            fps: Frames per second of the video
            
        Returns:
            bool: True if successful, False otherwise
        """
        import subprocess
        
        # Calculate start time and duration in seconds
        start_time = start_frame / fps
        duration = (end_frame - start_frame + 1) / fps
        
        # Build ffmpeg command
        # -ss: seek to start time (before input for fast seeking)
        # -i: input file
        # -t: duration to extract
        # -c:v copy: copy video codec (no re-encoding)
        # -avoid_negative_ts make_zero: fix timestamp issues
        cmd = [
            'ffmpeg',
            '-ss', str(start_time),  # Seek to start time
            '-i', str(video_path),   # Input file
            '-t', str(duration),     # Duration to extract
            '-c:v', 'copy',          # Copy video stream without re-encoding
            '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
            '-y',                    # Overwrite output file if exists
            str(output_path)
        ]
        
        print(f"Running ffmpeg command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("FFmpeg split completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error: {e.stderr}")
            return False


    def create_split_video_opencv(video_path, output_path, start_frame, end_frame):
        """
        Create a temporary video file containing only the specified frame range using OpenCV.
        This is the fallback method if ffmpeg is not available.
        
        Args:
            video_path: Path to the original video
            output_path: Path for the temporary split video
            start_frame: First frame to include
            end_frame: Last frame to include
            
        Returns:
            bool: True if successful, False otherwise
        """
        cap = cv2.VideoCapture(str(video_path))
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Define the codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Set position to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Write frames from start to end
        for frame_idx in range(start_frame, end_frame + 1):
            ret, frame = cap.read()
            if not ret:
                print(f"Failed to read frame {frame_idx}")
                break
            out.write(frame)
        
        # Release everything
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        return True


    def create_split_video(video_path, output_path, start_frame, end_frame, fps):
        """
        Create a temporary video file containing only the specified frame range.
        Tries ffmpeg first for speed, falls back to OpenCV if needed.
        
        Args:
            video_path: Path to the original video
            output_path: Path for the temporary split video
            start_frame: First frame to include
            end_frame: Last frame to include
            fps: Frames per second of the video
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if ffmpeg is available
        try:
            import subprocess
            subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
            use_ffmpeg = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("FFmpeg not found, falling back to OpenCV method")
            use_ffmpeg = False
        
        if use_ffmpeg:
            return create_split_video_ffmpeg(video_path, output_path, start_frame, end_frame, fps)
        else:
            return create_split_video_opencv(video_path, output_path, start_frame, end_frame)


    def clean_partial_outputs(video_path, split_index, total_splits, start_frame, end_frame):
        """
        Clean up any partial output files from previous failed runs.
        
        Args:
            video_path: Path to the original video
            split_index: Current split number
            total_splits: Total number of splits
            start_frame: Start frame for this split
            end_frame: End frame for this split
        """
        video_dir = video_path.parent
        video_stem = video_path.stem
        
        # Patterns for files that might exist from previous runs
        patterns = [
            # Split-specific patterns
            f"{video_stem}_split{split_index}of{total_splits}*",
            f"{video_stem}_frames{start_frame}-{end_frame}_split{split_index}of{total_splits}*",
            # Temporary/partial files
            f"{video_stem}*_split{split_index}_temp*",
            f"{video_stem}*_split{split_index}_partial*",
        ]
        
        print("Cleaning up potential partial files from previous runs...")
        for pattern in patterns:
            for file_path in video_dir.glob(pattern):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                        print(f"  Removed partial file: {file_path.name}")
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
                        print(f"  Removed partial directory: {file_path.name}")
                except Exception as e:
                    print(f"  Warning: Could not remove {file_path.name}: {e}")


    def analyse_split(video_path, gpu_id, split_index, total_splits):
        """
        Analyse a specific split of a video.
        
        Args:
            video_path: Path to the video file
            gpu_id: GPU index to use
            split_index: Current split number (1-based)
            total_splits: Total number of splits
        """
        start_time = time.perf_counter()
        video_path = Path(video_path)
        
        print(f"Analysing split {split_index}/{total_splits} of {video_path.name}")
        print(f"Using GPU {gpu_id}")
        
        # Get video information
        total_frames, fps = get_video_info(video_path)
        print(f"Video info: {total_frames} frames at {fps:.2f} FPS")
        
        # Calculate split boundaries
        start_frame, end_frame = calculate_split_frames(total_frames, split_index, total_splits)
        print(f"Processing frames {start_frame} to {end_frame} ({end_frame - start_frame + 1} frames)")
        
        # Clean up any partial files from previous runs
        clean_partial_outputs(video_path, split_index, total_splits, start_frame, end_frame)
        
        # Create temporary split video
        temp_dir = video_path.parent / "temp_splits"
        temp_dir.mkdir(exist_ok=True)
        
        temp_video_name = f"{video_path.stem}_split{split_index}of{total_splits}.avi"
        temp_video_path = temp_dir / temp_video_name
        
        # Remove temp video if it already exists
        if temp_video_path.exists():
            print(f"Removing existing temp video: {temp_video_path}")
            temp_video_path.unlink()
        
        print(f"Creating temporary split video: {temp_video_path}")
        if not create_split_video(video_path, temp_video_path, start_frame, end_frame, fps):
            raise RuntimeError("Failed to create split video")
        
        # Analyse the split video
        try:
            # Create a marker file to indicate analysis is in progress
            progress_marker = video_path.parent / f".{video_path.stem}_split{split_index}_in_progress"
            progress_marker.touch()
            
            analyze_videos(
                config=config,
                videos=[str(temp_video_path)], 
                videotype='.avi', 
                save_as_csv=True, 
                gputouse=gpu_id,
                destfolder=str(video_path.parent)  # Save outputs in original video directory
            )
            
            # Remove progress marker
            if progress_marker.exists():
                progress_marker.unlink()
            
            # Rename output files to include split information and frame range
            # Find the generated files
            base_pattern = temp_video_path.stem
            renamed_files = []
            
            for file in video_path.parent.glob(f"{base_pattern}*"):
                if file.suffix in ['.csv', '.h5', '.pickle']:
                    # Extract the DLC model name from the filename
                    dlc_suffix = file.name[len(base_pattern):]
                    
                    # Create new filename with frame range information
                    new_name = f"{video_path.stem}_frames{start_frame}-{end_frame}_split{split_index}of{total_splits}{dlc_suffix}"
                    new_path = file.parent / new_name
                    
                    # Remove target file if it already exists (from previous run)
                    if new_path.exists():
                        print(f"  Removing existing output file: {new_path.name}")
                        new_path.unlink()
                    
                    # Rename the file
                    file.rename(new_path)
                    renamed_files.append(new_name)
                    print(f"  Renamed output: {file.name} -> {new_name}")
            
            if not renamed_files:
                raise RuntimeError("No output files were generated by DeepLabCut")
        
        except Exception as e:
            # Clean up on error
            if 'progress_marker' in locals() and progress_marker.exists():
                progress_marker.unlink()
            
            # Clean up any partial output files
            print("Cleaning up partial outputs due to error...")
            for pattern in [f"{temp_video_path.stem}*", f"{video_path.stem}_split{split_index}*"]:
                for file in video_path.parent.glob(pattern):
                    if file != temp_video_path:  # Don't remove the temp video yet
                        try:
                            file.unlink()
                            print(f"  Removed partial output: {file.name}")
                        except Exception as cleanup_error:
                            print(f"  Warning: Could not remove {file.name}: {cleanup_error}")
            
            raise e
        
        finally:
            # Always clean up temporary video
            if temp_video_path.exists():
                temp_video_path.unlink()
                print(f"Removed temporary video: {temp_video_path}")
            
            # Clean up temp directory if empty
            if temp_dir.exists() and not any(temp_dir.iterdir()):
                temp_dir.rmdir()
                print(f"Removed empty temp directory: {temp_dir}")
        
        # Calculate and print timing
        elapsed_time = time.perf_counter() - start_time
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        
        print(f"""Analysis of split {split_index}/{total_splits} complete. 
                    \rTook: {minutes} minutes and {seconds:.2f} seconds""")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python batch_analyse_split.py VIDEO_PATH GPU_ID SPLIT_INDEX TOTAL_SPLITS")
        sys.exit(1)
    
    video_path = sys.argv[1]
    gpu_id = int(sys.argv[2])
    split_index = int(sys.argv[3])
    total_splits = int(sys.argv[4])
    
    try:
        analyse_split(video_path, gpu_id, split_index, total_splits)
    except Exception as e:
        print(f"FATAL ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)