#!/usr/bin/env python
"""
Advanced background subtraction for greyscale mouse tracking videos.
Optimised for cases where mouse and background have similar intensities.
Uses adaptive MOG2 with preprocessing enhancements.
"""

import os
import sys
import cv2
import numpy as np
import tempfile
import shutil
import subprocess
from pathlib import Path
from multiprocessing import Pool, cpu_count, current_process
import argparse
import time
from datetime import datetime
from tqdm import tqdm
import json
import logging


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(processName)s - %(message)s',
    datefmt='%H:%M:%S'
)


def print_progress(message, level="INFO"):
    """
    Print progress message with timestamp.
    
    Args:
        message (str): Message to print
        level (str): Message level (INFO, WARNING, ERROR, SUCCESS, DEBUG)
    """
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    colours = {
        "INFO": "\033[94m",
        "WARNING": "\033[93m",
        "ERROR": "\033[91m",
        "SUCCESS": "\033[92m",
        "DEBUG": "\033[95m",
        "RESET": "\033[0m"
    }
    
    colour = colours.get(level, colours["INFO"])
    reset = colours["RESET"]
    
    print(f"{colour}[{timestamp}] {level}: {message}{reset}")
    sys.stdout.flush()


def enhance_contrast_clahe(frame):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) to enhance local contrast.
    Particularly effective for distinguishing similar grey values.
    
    Args:
        frame (np.ndarray): Input frame (colour or greyscale)
        
    Returns:
        np.ndarray: Contrast enhanced frame
    """
    # Convert to greyscale if needed
    if len(frame.shape) == 3:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        grey = frame.copy()
    
    # Create CLAHE object with parameters tuned for mouse detection
    clahe = cv2.createCLAHE(
        clipLimit=3.0,      # Limit contrast amplification
        tileGridSize=(8, 8) # Size of grid for local histogram equalisation
    )
    
    # Apply CLAHE
    enhanced = clahe.apply(grey)
    
    # Convert back to colour if input was colour
    if len(frame.shape) == 3:
        return cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    
    return enhanced


def create_edge_enhanced_frame(frame, blur_size=5):
    """
    Enhance edges to help distinguish mouse from background.
    
    Args:
        frame (np.ndarray): Input frame
        blur_size (int): Gaussian blur kernel size
        
    Returns:
        np.ndarray: Edge-enhanced frame
    """
    # Convert to greyscale if needed
    if len(frame.shape) == 3:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        grey = frame
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(grey, (blur_size, blur_size), 0)
    
    # Calculate gradients
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    
    # Combine gradients
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    gradient_magnitude = np.uint8(np.clip(gradient_magnitude, 0, 255))
    
    return gradient_magnitude


def create_optimised_mog2():
    """
    Create MOG2 background subtractor optimised for mouse tracking.
    
    Returns:
        cv2.BackgroundSubtractorMOG2: Configured background subtractor
    """
    # Create MOG2 with parameters tuned for subtle greyscale differences
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=300,         # Shorter history for faster adaptation
        varThreshold=10,     # Lower threshold for sensitive detection
        detectShadows=True   # Important for mouse shadows
    )
    
    # Fine-tune parameters
    bg_subtractor.setBackgroundRatio(0.7)    # Less data for background
    bg_subtractor.setNMixtures(5)            # More Gaussian components
    bg_subtractor.setShadowValue(127)        # Shadow marking
    bg_subtractor.setShadowThreshold(0.5)    # Shadow detection threshold
    
    return bg_subtractor


def preprocess_frame(frame, enhance_contrast=True, enhance_edges=False):
    """
    Preprocess frame to improve mouse detection.
    
    Args:
        frame (np.ndarray): Input frame
        enhance_contrast (bool): Apply CLAHE contrast enhancement
        enhance_edges (bool): Apply edge enhancement
        
    Returns:
        np.ndarray: Preprocessed frame
    """
    result = frame.copy()
    
    if enhance_contrast:
        result = enhance_contrast_clahe(result)
    
    if enhance_edges:
        # Blend edge information with original
        edges = create_edge_enhanced_frame(result)
        if len(result.shape) == 3:
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        result = cv2.addWeighted(result, 0.7, edges, 0.3, 0)
    
    return result


def postprocess_mask(mask, min_area=100, fill_holes=True, smooth_gradient=False, fade_distance=30):
    """
    Clean up the foreground mask with morphological operations.
    
    Args:
        mask (np.ndarray): Binary mask
        min_area (int): Minimum contour area to keep
        fill_holes (bool): Whether to fill holes in contours
        smooth_gradient (bool): Whether to create smooth gradient mask
        fade_distance (int): Distance in pixels for gradient fade
        
    Returns:
        np.ndarray: Cleaned mask
    """
    # Remove shadows if present (MOG2 marks shadows as 127)
    mask[mask == 127] = 0
    
    # Ensure binary
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    
    # Morphological opening to remove noise
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by area and fill holes
    mask_clean = np.zeros_like(mask)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            cv2.drawContours(mask_clean, [contour], -1, 255, -1)
    
    # Morphological closing to smooth boundaries
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close)
    
    # Additional dilation to capture fuzzy edges
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_clean = cv2.dilate(mask_clean, kernel_dilate, iterations=1)
    
    # Apply smooth gradient if requested
    if smooth_gradient:
        mask_clean = create_smooth_gradient_mask(mask_clean, fade_distance)
    
    return mask_clean


def create_smooth_gradient_mask(mask, fade_distance=30):
    """
    Create a smooth gradient mask using distance transform.
    The mask fades from full intensity at the object to zero at fade_distance pixels away.
    
    Args:
        mask (np.ndarray): Binary mask (0 or 255)
        fade_distance (int): Distance in pixels for the fade effect
        
    Returns:
        np.ndarray: Gradient mask with values from 0-255
    """
    # Ensure binary mask
    binary_mask = (mask > 127).astype(np.uint8)
    
    # Calculate distance transform - distance from each pixel to nearest zero pixel
    dist_transform = cv2.distanceTransform(binary_mask, cv2.DIST_L2, 5)
    
    # Normalise distances to fade_distance
    # Pixels at distance 0 = 255 (full intensity)
    # Pixels at distance >= fade_distance = 0 (no intensity)
    gradient_mask = np.clip(dist_transform / fade_distance, 0, 1)
    
    # Apply a smooth falloff curve (quadratic for more natural fade)
    # You can experiment with different curves:
    # Linear: gradient_mask = gradient_mask
    # Quadratic (gentler): gradient_mask = gradient_mask ** 0.5
    # Cubic (sharper): gradient_mask = gradient_mask ** 2
    gradient_mask = gradient_mask ** 0.7  # Slightly gentler than linear
    
    # Convert back to 0-255 range
    gradient_mask = (gradient_mask * 255).astype(np.uint8)
    
    # Ensure original foreground pixels remain at full intensity
    gradient_mask = np.maximum(gradient_mask, binary_mask * 255)
    
    return gradient_mask


def process_chunk_mog2(chunk_info):
    """
    Process a video chunk using MOG2 with preprocessing.
    
    Args:
        chunk_info (tuple): Processing parameters
        
    Returns:
        tuple: Processing results
    """
    chunk_path, output_path, params, chunk_idx, total_chunks = chunk_info
    
    start_time = time.time()
    chunk_name = os.path.basename(chunk_path)
    
    logging.info(f"Processing chunk {chunk_idx + 1}/{total_chunks}: {chunk_name}")
    
    try:
        # Open video
        cap = cv2.VideoCapture(chunk_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video {chunk_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Setup output
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Create background subtractor
        bg_subtractor = create_optimised_mog2()
        
        # Learning phase - process initial frames at higher learning rate
        learning_frames = min(params.get('learning_frames', 50), total_frames // 2)
        logging.info(f"Chunk {chunk_idx + 1}: Learning phase for {learning_frames} frames")
        
        for i in range(learning_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess
            processed = preprocess_frame(
                frame,
                enhance_contrast=params.get('enhance_contrast', True),
                enhance_edges=params.get('enhance_edges', False)
            )
            
            # Learn background with higher rate initially
            bg_subtractor.apply(processed, learningRate=0.01)
        
        # Reset to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_count = 0
        
        # Process all frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Preprocess
            processed = preprocess_frame(
                frame,
                enhance_contrast=params.get('enhance_contrast', True),
                enhance_edges=params.get('enhance_edges', False)
            )
            
            # Apply background subtraction
            fg_mask = bg_subtractor.apply(processed, learningRate=params.get('learning_rate', 0.001))
            
            # Clean up mask
            fg_mask = postprocess_mask(
                fg_mask,
                min_area=params.get('min_area', 100),
                fill_holes=True,
                smooth_gradient=params.get('smooth_gradient', False),
                fade_distance=params.get('fade_distance', 30)
            )
            
            # Create output frame
            result = frame.copy()
            
            # Apply mask based on output mode
            if params.get('output_mode', 'highlight') == 'highlight':
                # For gradient masks, we need to handle differently
                if params.get('smooth_gradient', False):
                    # Convert gradient mask to 0-1 range
                    mask_norm = fg_mask.astype(np.float32) / 255.0
                    
                    # Create darkened background
                    bg_dark = (frame * params.get('bg_darkness', 0)).astype(np.uint8)
                    
                    # Blend based on gradient mask
                    for c in range(3 if len(frame.shape) == 3 else 1):
                        if len(frame.shape) == 3:
                            result[:, :, c] = (frame[:, :, c] * mask_norm + 
                                              bg_dark[:, :, c] * (1 - mask_norm)).astype(np.uint8)
                        else:
                            result = (frame * mask_norm + bg_dark * (1 - mask_norm)).astype(np.uint8)
                else:
                    # Original binary mask behaviour
                    result[fg_mask == 0] = (result[fg_mask == 0] * params.get('bg_darkness', 0.3)).astype(np.uint8)
                
                # Optional: Add coloured overlay on detected mouse
                if params.get('highlight_mouse', False):
                    overlay = result.copy()
                    overlay[fg_mask > 128] = [0, 255, 0] if len(result.shape) == 3 else 255
                    result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
                    
            elif params.get('output_mode', 'highlight') == 'mask_only':
                # Output just the mask
                result = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR) if len(result.shape) == 3 else fg_mask
                
            elif params.get('output_mode', 'highlight') == 'cutout':
                # For gradient mask with cutout
                if params.get('smooth_gradient', False):
                    mask_norm = fg_mask.astype(np.float32) / 255.0
                    for c in range(3 if len(frame.shape) == 3 else 1):
                        if len(frame.shape) == 3:
                            result[:, :, c] = (frame[:, :, c] * mask_norm).astype(np.uint8)
                        else:
                            result = (frame * mask_norm).astype(np.uint8)
                else:
                    result[fg_mask == 0] = 0
            
            # Optional: Draw contours
            if params.get('draw_contours', False):
                contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
            
            out.write(result)
            frame_count += 1
            
            if frame_count % 100 == 0:
                logging.info(f"Chunk {chunk_idx + 1}: Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        out.release()
        
        processing_time = time.time() - start_time
        logging.info(f"Completed chunk {chunk_idx + 1}: {frame_count} frames in {processing_time:.2f}s")
        
        return output_path, frame_count, processing_time, chunk_name, chunk_idx
        
    except Exception as e:
        logging.error(f"Error processing chunk {chunk_idx + 1}: {str(e)}")
        raise


def get_video_info(video_path):
    """Get video information using ffprobe."""
    cmd = [
        'ffprobe', '-v', 'quiet', '-print_format', 'json',
        '-show_format', '-show_streams', str(video_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    
    video_stream = next(s for s in data['streams'] if s['codec_type'] == 'video')
    fps_parts = video_stream['r_frame_rate'].split('/')
    fps = float(fps_parts[0]) / float(fps_parts[1])
    
    return {
        'fps': fps,
        'duration': float(data['format']['duration']),
        'frame_count': int(video_stream['nb_frames']),
        'width': int(video_stream['width']),
        'height': int(video_stream['height'])
    }


def split_video_ffmpeg(video_path, output_dir, num_chunks):
    """Split video into chunks using ffmpeg."""
    video_info = get_video_info(video_path)
    total_duration = video_info['duration']
    chunk_duration = total_duration / num_chunks
    
    chunk_paths = []
    
    with tqdm(total=num_chunks, desc="Creating chunks", unit="chunk") as pbar:
        for i in range(num_chunks):
            start_time = i * chunk_duration
            duration = chunk_duration if i < num_chunks - 1 else total_duration - start_time
            
            chunk_path = os.path.join(output_dir, f"chunk_{i:04d}.avi")
            
            cmd = [
                'ffmpeg', '-ss', str(start_time), '-i', str(video_path),
                '-t', str(duration), '-c:v', 'copy', '-v', 'quiet', '-y', chunk_path
            ]
            
            subprocess.run(cmd, check=True)
            chunk_paths.append(chunk_path)
            pbar.update(1)
    
    return chunk_paths


def combine_chunks_ffmpeg(chunk_paths, output_path):
    """Combine chunks back into single video."""
    concat_file = output_path.replace('.avi', '_concat.txt')
    
    with open(concat_file, 'w') as f:
        for chunk_path in chunk_paths:
            f.write(f"file '{os.path.abspath(chunk_path)}'\n")
    
    cmd = [
        'ffmpeg', '-f', 'concat', '-safe', '0', '-i', concat_file,
        '-c:v', 'libx264', '-preset', 'medium', '-crf', '18',
        '-v', 'quiet', '-y', output_path
    ]
    
    subprocess.run(cmd, check=True)
    os.remove(concat_file)


def initialise_worker():
    """Initialise worker process."""
    cv2.setNumThreads(1)


def process_mouse_video(video_path, output_path=None, params=None, num_processes=None):
    """
    Main function to process mouse tracking video with advanced background subtraction.
    
    Args:
        video_path (str): Path to input video
        output_path (str): Path for output video (auto-generated if None)
        params (dict): Processing parameters
        num_processes (int): Number of parallel processes
        
    Returns:
        str: Path to output video
    """
    # Default parameters optimised for greyscale mouse tracking
    default_params = {
        'enhance_contrast': True,      # Apply CLAHE enhancement
        'enhance_edges': False,        # Edge enhancement (set True if mouse has clear edges)
        'learning_frames': 50,         # Frames for initial learning
        'learning_rate': 0.001,        # Ongoing learning rate
        'min_area': 100,              # Minimum blob area (adjust based on mouse size)
        'bg_darkness': 0.3,           # How dark to make background
        'output_mode': 'highlight',    # 'highlight', 'cutout', or 'mask_only'
        'draw_contours': False,       # Draw green contours
        'highlight_mouse': False,     # Add colour overlay on mouse
        'smooth_gradient': False,     # Use smooth gradient mask
        'fade_distance': 30          # Pixel distance for gradient fade
    }
    
    if params:
        default_params.update(params)
    params = default_params
    
    # Auto-generate output path
    if output_path is None:
        input_dir = os.path.dirname(video_path)
        input_name = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(input_dir, f"{input_name}_mog2_processed.avi")
    
    if num_processes is None:
        num_processes = min(cpu_count(), 150)
    
    print_progress("=" * 60)
    print_progress("MOUSE TRACKING BACKGROUND SUBTRACTION", "SUCCESS")
    print_progress("=" * 60)
    print_progress(f"Input: {video_path}")
    print_progress(f"Output: {output_path}")
    print_progress(f"Processors: {num_processes}")
    print_progress("Parameters:")
    for key, value in params.items():
        print_progress(f"  {key}: {value}")
    print_progress("=" * 60)
    
    overall_start = time.time()
    
    # Create temporary directory
    temp_dir = os.path.join(os.path.dirname(video_path), f"mouse_bg_temp_{int(time.time())}")
    chunks_dir = os.path.join(temp_dir, 'chunks')
    processed_dir = os.path.join(temp_dir, 'processed')
    os.makedirs(chunks_dir)
    os.makedirs(processed_dir)
    
    try:
        # Split video
        print_progress("Splitting video into chunks...")
        chunk_paths = split_video_ffmpeg(video_path, chunks_dir, num_processes)
        
        # Prepare processing arguments
        chunk_args = []
        for i, chunk_path in enumerate(chunk_paths):
            output_chunk = os.path.join(processed_dir, f"processed_{i:04d}.avi")
            chunk_args.append((chunk_path, output_chunk, params, i, len(chunk_paths)))
        
        # Process in parallel
        print_progress("Processing chunks in parallel...")
        processed_paths = []
        total_frames = 0
        
        with Pool(num_processes, initializer=initialise_worker) as pool:
            with tqdm(total=len(chunk_args), desc="Processing", unit="chunk") as pbar:
                for result in pool.imap_unordered(process_chunk_mog2, chunk_args):
                    path, frames, time_taken, name, idx = result
                    processed_paths.append((idx, path))
                    total_frames += frames
                    pbar.update(1)
        
        # Sort and combine
        processed_paths.sort(key=lambda x: x[0])
        processed_paths = [p[1] for p in processed_paths]
        
        print_progress("Combining processed chunks...")
        combine_chunks_ffmpeg(processed_paths, output_path)
        
        # Report statistics
        total_time = time.time() - overall_start
        print_progress("=" * 60, "SUCCESS")
        print_progress("PROCESSING COMPLETE!", "SUCCESS")
        print_progress(f"Total time: {total_time:.1f}s")
        print_progress(f"Frames processed: {total_frames}")
        print_progress(f"Processing rate: {total_frames/total_time:.1f} fps")
        print_progress(f"Output saved to: {output_path}")
        print_progress("=" * 60, "SUCCESS")
        
        return output_path
        
    finally:
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(
        description='Advanced background subtraction for greyscale mouse tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python mouse_bg_subtraction.py video.avi
  python mouse_bg_subtraction.py video.avi --enhance_edges --min_area 50
  python mouse_bg_subtraction.py video.avi --output_mode cutout --num_processes 8
  python mouse_bg_subtraction.py video.avi --smooth_gradient --fade_distance 40
  
Output modes:
  highlight: Darken background, keep mouse bright (default)
  cutout: Black background, mouse only
  mask_only: Output the binary mask

Smooth gradient mask:
  --smooth_gradient: Creates a smooth fade from mouse to background
  --fade_distance: How many pixels the fade extends (default: 30)
        """
    )
    
    parser.add_argument('input_video', help='Path to input video file')
    parser.add_argument('--output', help='Output video path (auto-generated if not specified)')
    parser.add_argument('--enhance_contrast', action='store_true', default=True,
                       help='Apply CLAHE contrast enhancement (default: True)')
    parser.add_argument('--enhance_edges', action='store_true',
                       help='Apply edge enhancement')
    parser.add_argument('--learning_frames', type=int, default=50,
                       help='Initial learning frames (default: 50)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Background learning rate (default: 0.001)')
    parser.add_argument('--min_area', type=int, default=100,
                       help='Minimum blob area in pixels (default: 100)')
    parser.add_argument('--bg_darkness', type=float, default=0,
                       help='Background darkness 0-1 (default: 0.3)')
    parser.add_argument('--output_mode', choices=['highlight', 'cutout', 'mask_only'],
                       default='highlight', help='Output visualisation mode')
    parser.add_argument('--draw_contours', action='store_true',
                       help='Draw contours around detected objects')
    parser.add_argument('--highlight_mouse', action='store_true',
                       help='Add colour overlay on detected mouse')
    parser.add_argument('--smooth_gradient', action='store_true',
                       help='Use smooth gradient mask instead of binary')
    parser.add_argument('--fade_distance', type=int, default=30,
                       help='Pixel distance for gradient fade (default: 30)')
    parser.add_argument('--num_processes', type=int, default=None,
                       help='Number of parallel processes (default: auto)')
    
    args = parser.parse_args()
    
    # Check input exists
    if not os.path.exists(args.input_video):
        print_progress(f"Input video not found: {args.input_video}", "ERROR")
        sys.exit(1)
    
    # Build parameters from arguments
    params = {
        'enhance_contrast': args.enhance_contrast,
        'enhance_edges': args.enhance_edges,
        'learning_frames': args.learning_frames,
        'learning_rate': args.learning_rate,
        'min_area': args.min_area,
        'bg_darkness': args.bg_darkness,
        'output_mode': args.output_mode,
        'draw_contours': args.draw_contours,
        'highlight_mouse': args.highlight_mouse,
        'smooth_gradient': args.smooth_gradient,
        'fade_distance': args.fade_distance
    }
    
    try:
        output_path = process_mouse_video(
            args.input_video,
            output_path=args.output,
            params=params,
            num_processes=args.num_processes
        )
        
        print_progress("Ready for DeepLabCut analysis!", "SUCCESS")
        
    except Exception as e:
        print_progress(f"Processing failed: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()