"""
DeepLabCut Performance Evaluation Script

This script evaluates DeepLabCut analysis performance by generating annotated images
showing all tracked body parts, head angles, LED positions, and calculated angles at cue onset.
Images can be sorted by cue presentation angle or mouse head bearing.
"""

import cv2 as cv
import numpy as np
import math
from pathlib import Path
import os
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm


def evaluate_deeplabcut_performance(
    session, 
    output_directory: str,
    likelihood_threshold: float = 0.6,
    sort_by: str = 'cue_angle',
    frame_offset: int = 0  # Add this parameter
) -> None:
    """
    Evaluate DeepLabCut performance by creating annotated images at cue onset.
    
    This function processes each trial in the session, finds the frame at cue onset,
    and creates annotated images showing:
    - All tracked body parts with likelihood indicators
    - Calculated head angle line
    - LED position
    - Angle values and likelihoods as text overlays
    
    Images are saved in order sorted by either cue presentation angle or mouse head bearing.
    
    Parameters:
    -----------
    session : Session object
        Loaded session object containing trial data, DLC coordinates, and video information
    output_directory : str
        Path where annotated images will be saved
    likelihood_threshold : float, default=0.6
        Minimum likelihood threshold for ear position detection
    sort_by : str, default='cue_angle'
        Sorting method for output images. Options:
        - 'cue_angle': Sort by cue presentation angle
        - 'head_bearing': Sort by mouse head bearing at cue onset
    frame_offset : int, default=0
        Number of frames to offset the displayed video frame relative to cue onset.
        Positive values show frames after cue onset, negative values show frames before.
        DeepLabCut data will still be from the original cue onset frame.

    Returns:
    --------
    None
        Images are saved to the specified output directory
    
    Raises:
    -------
    ValueError
        If sort_by parameter is not 'cue_angle' or 'head_bearing'
    """
    
    # Validate sort_by parameter
    if sort_by not in ['cue_angle', 'head_bearing']:
        raise ValueError("sort_by must be either 'cue_angle' or 'head_bearing'")
    
    # Create output directory structure
    session_id = session.session_ID
    output_folder_name = f"{session_id}_dlc_evaluation_sorted_by_{sort_by}"
    output_path = Path(output_directory) / output_folder_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Count total trials for progress tracking
    total_trials = len(session.trials)
    print(f"Processing {total_trials} trials for session {session_id}")
    print(f"Sorting method: {sort_by}")
    
    # First pass: collect all valid trials with their data
    print("\nPhase 1: Collecting valid trials...")
    valid_trials = []
    skipped_trials = []
    
    with tqdm(total=total_trials, desc="Collecting trials", unit="trial") as pbar:
        for trial_idx, trial in enumerate(session.trials):
            try:
                # Update progress bar description with current trial info
                pbar.set_description(f"Validating trial {trial_idx + 1}/{total_trials}")
                
                # Skip trials without required data or turn_data
                if not _validate_trial_data(trial):
                    skipped_trials.append({
                        'trial_idx': trial_idx,
                        'reason': 'Missing required data or catch trial'
                    })
                    pbar.update(1)
                    continue
                    
                # Skip trials with low likelihood ear detections
                if not _validate_ear_likelihoods(trial, likelihood_threshold):
                    skipped_trials.append({
                        'trial_idx': trial_idx,
                        'reason': f'Ear likelihood below threshold ({likelihood_threshold})'
                    })
                    pbar.update(1)
                    continue
                    
                # Find cue onset frame
                cue_onset_frame = _find_cue_onset_frame(trial)
                if cue_onset_frame is None:
                    skipped_trials.append({
                        'trial_idx': trial_idx,
                        'reason': 'Could not find cue onset frame'
                    })
                    pbar.update(1)
                    continue
                    
                # Collect trial data for sorting
                turn_data = trial['turn_data']
                cue_angle = abs(turn_data['cue_presentation_angle'])
                head_bearing = turn_data['bearing']
                
                valid_trials.append({
                    'trial_idx': trial_idx,
                    'trial': trial,
                    'cue_onset_frame': cue_onset_frame,
                    'cue_angle': cue_angle,
                    'head_bearing': head_bearing,
                    'sort_value': cue_angle if sort_by == 'cue_angle' else head_bearing
                })
                
                # Update progress bar with collection info
                pbar.set_postfix({
                    'collected': len(valid_trials),
                    'skipped': len(skipped_trials)
                })
                
            except Exception as e:
                skipped_trials.append({
                    'trial_idx': trial_idx,
                    'reason': f'Error during validation: {str(e)}'
                })
                
            # Update progress bar
            pbar.update(1)
    
    # Sort valid trials by the specified parameter
    print(f"\nPhase 2: Sorting {len(valid_trials)} valid trials by {sort_by}...")
    valid_trials.sort(key=lambda x: x['sort_value'])
    
    # Second pass: process sorted trials and create images
    print("\nPhase 3: Processing sorted trials and creating images...")
    processed_trials = []
    processing_errors = []
    
    with tqdm(total=len(valid_trials), desc="Creating images", unit="image") as pbar:
        for sorted_idx, trial_data in enumerate(valid_trials):
            try:
                trial_idx = trial_data['trial_idx']
                trial = trial_data['trial']
                cue_onset_frame = trial_data['cue_onset_frame']
                cue_angle = trial_data['cue_angle']
                head_bearing = trial_data['head_bearing']
                
                # Update progress bar description
                pbar.set_description(
                    f"Processing sorted position {sorted_idx + 1}/{len(valid_trials)} "
                    f"(original trial {trial_idx})"
                )
                
                # Load video frame using session's video path
                offset_frame_number = cue_onset_frame + frame_offset
                video_frame = _load_video_frame(session, offset_frame_number)
                if video_frame is None:
                    processing_errors.append({
                        'trial_idx': trial_idx,
                        'sorted_idx': sorted_idx,
                        'reason': 'Could not load video frame'
                    })
                    pbar.update(1)
                    continue
                    
                # Create annotated image using existing session calculations
                annotated_frame = _create_annotated_image(
                    video_frame, trial, session, cue_onset_frame, frame_offset
                )
                
                # Save annotated image with sorted index in filename
                # Include both cue angle and head bearing in filename for reference
                filename = (
                    f"{sorted_idx:04d}_trial{trial_idx:03d}_"
                    f"cue{cue_angle:05.1f}_bearing{head_bearing:05.1f}.jpg"
                )
                output_filepath = output_path / filename
                cv.imwrite(str(output_filepath), annotated_frame)
                
                processed_trials.append({
                    'sorted_idx': sorted_idx,
                    'trial_idx': trial_idx,
                    'cue_angle': cue_angle,
                    'head_bearing': head_bearing,
                    'filename': filename
                })
                
                # Update progress bar with success info
                pbar.set_postfix({
                    'processed': len(processed_trials),
                    'errors': len(processing_errors),
                    f'current_{sort_by}': f"{trial_data['sort_value']:.1f}°"
                })
                
            except Exception as e:
                processing_errors.append({
                    'trial_idx': trial_idx,
                    'sorted_idx': sorted_idx,
                    'reason': f'Error during processing: {str(e)}'
                })
                
            # Update progress bar
            pbar.update(1)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"Total trials: {total_trials}")
    print(f"Valid trials found: {len(valid_trials)}")
    print(f"Successfully processed: {len(processed_trials)}")
    print(f"Skipped during validation: {len(skipped_trials)}")
    print(f"Errors during processing: {len(processing_errors)}")
    print(f"Overall success rate: {len(processed_trials)/total_trials*100:.1f}%")
    print(f"Images sorted by: {sort_by}")
    print(f"Images saved to: {output_path}")
    
    # Print sorting range information
    if processed_trials:
        if sort_by == 'cue_angle':
            min_val = min(t['cue_angle'] for t in processed_trials)
            max_val = max(t['cue_angle'] for t in processed_trials)
            print(f"\nCue angle range: {min_val:.1f}° to {max_val:.1f}°")
        else:
            min_val = min(t['head_bearing'] for t in processed_trials)
            max_val = max(t['head_bearing'] for t in processed_trials)
            print(f"\nHead bearing range: {min_val:.1f}° to {max_val:.1f}°")
    
    # Print details of skipped trials if any
    if skipped_trials:
        print(f"\nSkipped trials details (first 10):")
        print(f"-" * 40)
        for skipped in skipped_trials[:10]:
            print(f"Trial {skipped['trial_idx']:3d}: {skipped['reason']}")
        if len(skipped_trials) > 10:
            print(f"... and {len(skipped_trials) - 10} more")
    
    # Print processing errors if any
    if processing_errors:
        print(f"\nProcessing errors (first 10):")
        print(f"-" * 40)
        for error in processing_errors[:10]:
            print(f"Trial {error['trial_idx']:3d} (sorted position {error['sorted_idx']}): {error['reason']}")
        if len(processing_errors) > 10:
            print(f"... and {len(processing_errors) - 10} more")
    
    # Save processing summary
    _save_processing_summary(processed_trials, skipped_trials, processing_errors, output_path, sort_by)


def _validate_trial_data(trial: Dict) -> bool:
    """
    Validate that trial contains required data for processing.
    
    Parameters:
    -----------
    trial : Dict
        Trial dictionary containing trial data
        
    Returns:
    --------
    bool
        True if trial has required data, False otherwise
    """
    required_keys = ['video_frames', 'DLC_data', 'cue_start', 'correct_port', 'turn_data']
    
    for key in required_keys:
        if key not in trial or trial[key] is None:
            return False
    
    if len(trial['video_frames']) == 0:
        return False
    
    # Skip catch trials if they exist (optional - depends on whether you want to evaluate them)
    if trial.get('catch', False):
        return False
        
    return True


def _validate_ear_likelihoods(trial: Dict, likelihood_threshold: float) -> bool:
    """
    Validate that ear likelihoods meet the threshold using existing turn_data.
    
    Parameters:
    -----------
    trial : Dict
        Trial dictionary containing turn_data with likelihood information
    likelihood_threshold : float
        Minimum likelihood threshold
        
    Returns:
    --------
    bool
        True if both ear likelihoods meet threshold, False otherwise
    """
    if 'turn_data' not in trial or trial['turn_data'] is None:
        return False
        
    turn_data = trial['turn_data']
    
    left_likelihood = turn_data.get('left_ear_likelihood', 0.0)
    right_likelihood = turn_data.get('right_ear_likelihood', 0.0)
    
    return left_likelihood >= likelihood_threshold and right_likelihood >= likelihood_threshold


def _find_cue_onset_frame(trial: Dict) -> Optional[int]:
    """
    Find the video frame number corresponding to cue onset.
    
    Parameters:
    -----------
    trial : Dict
        Trial dictionary containing timestamps and video frame data
        
    Returns:
    --------
    Optional[int]
        Frame number at cue onset, or None if not found
    """
    try:
        dlc_data = trial['DLC_data']
        cue_start_time = trial['cue_start']
        
        # Find frame closest to cue onset time
        timestamps = dlc_data['timestamps'].values
        frame_index = np.searchsorted(timestamps, cue_start_time, side='left')
        
        # Ensure we have a valid frame index
        if frame_index >= len(timestamps):
            frame_index = len(timestamps) - 1
        elif frame_index > 0 and timestamps[frame_index] > cue_start_time:
            frame_index -= 1
            
        # Get the corresponding video frame number
        video_frames = trial['video_frames']
        if frame_index < len(video_frames):
            return video_frames[frame_index]
            
    except (KeyError, IndexError, TypeError):
        pass
        
    return None


def _load_video_frame(session, frame_number: int) -> Optional[np.ndarray]:
    """
    Load a specific frame from the session video using session's video path.
    
    Parameters:
    -----------
    session : Session object
        Session object containing video file path
    frame_number : int
        Frame number to load
        
    Returns:
    --------
    Optional[np.ndarray]
        Video frame as numpy array, or None if loading fails
    """
    try:
        # Use session's video path
        video_path = session.session_video
        
        # Load video frame
        cap = cv.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        
        # Get total frame count for bounds checking
        total_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        
        # Ensure frame number is within bounds
        if frame_number < 0 or frame_number >= total_frames:
            cap.release()
            return None
            
        cap.set(cv.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return frame
            
    except Exception:
        pass
        
    return None


def _get_body_part_colours() -> Dict[str, Tuple[int, int, int]]:
    """
    Define distinct colours for each body part.
    
    Returns:
    --------
    Dict[str, Tuple[int, int, int]]
        Dictionary mapping body part names to BGR colour tuples
    """
    return {
        'left_ear': (0, 255, 0),        # Green
        'right_ear': (255, 0, 0),        # Blue
        'nose': (0, 255, 255),           # Yellow
        'head': (255, 0, 255),           # Magenta
        'base_neck': (255, 128, 0),      # Orange
        'body': (128, 0, 255),           # Purple
        'tail_base': (0, 128, 255),      # Light Blue
        'tail_mid': (255, 128, 128),     # Pink
        'tail_end': (128, 255, 128),     # Light Green
        'spine1': (255, 200, 0),         # Gold
        'spine2': (200, 255, 0),         # Lime
        'spine3': (0, 200, 255),         # Cyan
        'spine4': (255, 0, 200),         # Hot Pink
        # Add more body parts and colours as needed
    }


def _create_annotated_image(
    frame: np.ndarray, 
    trial: Dict, 
    session,
    cue_onset_frame: int,
    frame_offset: int = 0  # Add this parameter
) -> np.ndarray:
    """
    Create annotated image with all body parts, head angle, and text overlays.
    Uses existing session calculations to avoid recalculation.
    
    Parameters:
    -----------
    frame : np.ndarray
        Original video frame
    trial : Dict
        Trial dictionary containing turn_data with pre-calculated values
    session : Session object
        Session object containing port coordinates and other calculated values
    cue_onset_frame : int
        Frame number at cue onset
        
    Returns:
    --------
    np.ndarray
        Annotated image
    """
    # Create a copy of the frame for annotation
    annotated_frame = frame.copy()
    
    # Define colours
    head_line_colour = (255, 255, 255)  # White for head angle line
    led_colour = (0, 255, 255)  # Yellow for LED
    text_colour = (255, 255, 255)  # White for text
    success_colour = (0, 255, 0)  # Green for successful trials
    fail_colour = (0, 0, 255)  # Red for failed trials
    corrected_angle_colour = (255, 165, 0)  # Orange for corrected angles
    
    # Get body part colours
    body_part_colours = _get_body_part_colours()
    
    # Get all body part positions and likelihoods at cue onset
    all_body_parts = _get_all_body_parts_at_frame(trial, cue_onset_frame)
    
    if all_body_parts is not None:
        # Draw all body parts
        for body_part, data in all_body_parts.items():
            if body_part in body_part_colours:
                colour = body_part_colours[body_part]
            else:
                # Default colour for unlisted body parts
                colour = (128, 128, 128)  # Grey
            
            # Draw body part
            position = (int(data['x']), int(data['y']))
            cv.circle(annotated_frame, position, 8, colour, -1)
            
            # Draw outline if likelihood is low
            if data['likelihood'] < 0.6:
                cv.circle(annotated_frame, position, 10, (0, 0, 0), 2)
    
    # Get pre-calculated data from turn_data
    turn_data = trial['turn_data']
    head_midpoint = turn_data['midpoint']
    bearing = turn_data['bearing']
    cue_presentation_angle = turn_data['cue_presentation_angle']
    
    # Check if angle was corrected
    angle_correction_method = turn_data.get('angle_correction_method', 'none')
    angle_was_corrected = angle_correction_method != 'none'
    
    # Use different colour for head line if angle was corrected
    if angle_was_corrected:
        head_line_colour = corrected_angle_colour
    
    # Draw head angle line using pre-calculated bearing and midpoint
    head_midpoint_px = (int(head_midpoint[0]), int(head_midpoint[1]))
    
    # Calculate nose position using the bearing from turn_data
    nose_length = 60  # Length of the head direction line
    bearing_rad = math.radians(bearing)
    nose_x = int(head_midpoint[0] + nose_length * math.cos(bearing_rad))
    nose_y = int(head_midpoint[1] - nose_length * math.sin(bearing_rad))
    nose_position = (nose_x, nose_y)
    
    cv.line(annotated_frame, head_midpoint_px, nose_position, head_line_colour, 3)
    cv.circle(annotated_frame, head_midpoint_px, 5, head_line_colour, -1)
    
    # Draw LED position using session's pre-calculated port coordinates
    correct_port = trial['correct_port']
    if correct_port == "audio-1":
        port_index = 0
    else:
        port_index = int(correct_port) - 1
    
    # Use session's pre-calculated port coordinates
    if hasattr(session, 'port_coordinates') and port_index < len(session.port_coordinates):
        led_position = session.port_coordinates[port_index]
        cv.circle(annotated_frame, led_position, 10, led_colour, -1)
    
    # Draw trial outcome indicator
    trial_success = trial.get('success', False)
    outcome_colour = success_colour if trial_success else fail_colour
    outcome_text = "SUCCESS" if trial_success else "FAIL"
    
    # Draw outcome indicator in top-right corner
    frame_height, frame_width = annotated_frame.shape[:2]
    outcome_pos = (frame_width - 150, 40)
    cv.putText(annotated_frame, outcome_text, outcome_pos, cv.FONT_HERSHEY_SIMPLEX, 0.8, outcome_colour, 2)
    
    # Add text annotations including all body part likelihoods
    _add_text_annotations(
        annotated_frame, 
        trial,
        cue_presentation_angle,
        bearing,
        all_body_parts,
        text_colour,
        body_part_colours,
        turn_data,
        frame_offset
    )
    
    # Add legend for body part colours
    _add_body_part_legend(annotated_frame, body_part_colours, all_body_parts)
    
    return annotated_frame


def _get_all_body_parts_at_frame(trial: Dict, frame_number: int) -> Optional[Dict]:
    """
    Get all body part positions and likelihoods at a specific frame from DLC data.
    
    Parameters:
    -----------
    trial : Dict
        Trial dictionary containing DLC data
    frame_number : int
        Frame number to get positions for
        
    Returns:
    --------
    Optional[Dict]
        Dictionary with all body part positions and likelihoods, or None if not available
    """
    try:
        dlc_data = trial['DLC_data']
        video_frames = trial['video_frames']
        
        # Find index of the frame in video_frames list
        frame_idx = video_frames.index(frame_number)
        
        # Get DLC data at that index
        frame_data = dlc_data.iloc[frame_idx]
        
        # Extract all body parts
        body_parts = {}
        
        # Get unique body part names from the multi-index columns
        if hasattr(dlc_data.columns, 'levels'):
            body_part_names = dlc_data.columns.levels[0]
        else:
            # Handle case where columns might be structured differently
            body_part_names = set([col[0] for col in dlc_data.columns if isinstance(col, tuple)])
        
        for body_part in body_part_names:
            try:
                body_parts[body_part] = {
                    'x': frame_data[(body_part, 'x')],
                    'y': frame_data[(body_part, 'y')],
                    'likelihood': frame_data[(body_part, 'likelihood')]
                }
            except KeyError:
                # Skip if this body part doesn't have all coordinates
                continue
        
        return body_parts
        
    except (ValueError, IndexError, KeyError):
        return None


def _add_text_annotations(
    frame: np.ndarray, 
    trial: Dict,
    cue_presentation_angle: float,
    bearing: float,
    all_body_parts: Optional[Dict],
    text_colour: Tuple[int, int, int],
    body_part_colours: Dict[str, Tuple[int, int, int]],
    turn_data: Dict,
    frame_offset: int = 0  # Add this parameter
) -> None:
    """
    Add text annotations to the frame including all body part likelihoods and angle correction info.
    
    Parameters:
    -----------
    frame : np.ndarray
        Frame to annotate
    trial : Dict
        Trial dictionary containing trial information
    cue_presentation_angle : float
        Pre-calculated cue presentation angle
    bearing : float
        Pre-calculated head bearing
    all_body_parts : Optional[Dict]
        Dictionary of all body part positions and likelihoods
    text_colour : Tuple[int, int, int]
        Text colour in BGR format
    body_part_colours : Dict[str, Tuple[int, int, int]]
        Dictionary of body part colours
    turn_data : Dict
        Turn data containing angle correction information
    """
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    
    # Get angle correction information
    angle_correction_method = turn_data.get('angle_correction_method', 'none')
    ear_distance = turn_data.get('ear_distance', 'N/A')
    spine_data_available = turn_data.get('spine_data_available', False)
    
    # Text content using pre-calculated values and trial information
    texts = [
        f"Trial: {trial.get('trial_no', 'N/A')}",
        f"Port: {trial['correct_port']}",
        f"Phase: {trial.get('phase', 'N/A')}",
        f"Cue Angle: {cue_presentation_angle:.1f}°",
        f"Head Bearing: {bearing:.1f}°",
    ]

    # Add frame offset information if not zero
    if frame_offset != 0:
        offset_text = f"Frame Offset: {'+' if frame_offset > 0 else ''}{frame_offset}"
        texts.append(offset_text)
    
    # Add angle correction information
    if angle_correction_method != 'none':
        texts.append("")  # Empty line for spacing
        texts.append("ANGLE CORRECTION APPLIED:")
        
        if angle_correction_method == 'ear_flip_corrected':
            texts.append("  Method: Ear flip corrected")
        elif angle_correction_method == 'spine_based':
            texts.append("  Method: Spine-based calculation")
        elif angle_correction_method == 'ears_too_close_no_spine':
            texts.append("  Method: Ears too close (no spine)")
        
        if isinstance(ear_distance, (int, float)):
            texts.append(f"  Ear distance: {ear_distance:.1f} pixels")
        texts.append(f"  Spine data: {'Available' if spine_data_available else 'Not available'}")
    
    texts.extend([
        "",  # Empty line for spacing
        "Body Part Likelihoods:"
    ])
    
    # Add body part likelihoods
    if all_body_parts:
        for body_part, data in sorted(all_body_parts.items()):
            likelihood_text = f"{body_part}: {data['likelihood']:.3f}"
            texts.append(likelihood_text)
    
    # Check if it's a catch trial
    if trial.get('catch', False):
        texts.insert(0, "CATCH TRIAL")
    
    # Text positions (top-left corner with spacing)
    text_y_start = 30
    text_y_spacing = 25
    text_x = 20
    
    for i, text in enumerate(texts):
        if text:  # Only draw non-empty text
            y_pos = text_y_start + (i * text_y_spacing)
            
            # Get text size for background rectangle
            (text_width, text_height), baseline = cv.getTextSize(
                text, font, font_scale, thickness
            )
            
            # Determine text colour
            if text == "CATCH TRIAL":
                current_text_colour = (0, 165, 255)  # Orange for catch trials
            elif text == "ANGLE CORRECTION APPLIED:":
                current_text_colour = (255, 165, 0)  # Orange for correction header
            elif text.startswith("  Method:") or text.startswith("  Ear distance:") or text.startswith("  Spine data:"):
                current_text_colour = (255, 165, 0)  # Orange for correction details
            elif text.startswith("Body Part Likelihoods"):
                current_text_colour = text_colour
            elif ":" in text and any(text.startswith(bp + ":") for bp in all_body_parts.keys() if all_body_parts):
                # Use body part colour for likelihood text
                body_part_name = text.split(":")[0]
                if body_part_name in body_part_colours:
                    current_text_colour = body_part_colours[body_part_name]
                else:
                    current_text_colour = (128, 128, 128)  # Grey for unlisted parts
            else:
                current_text_colour = text_colour
            
            # Draw background rectangle
            cv.rectangle(
                frame, 
                (text_x - 5, y_pos - text_height - 5), 
                (text_x + text_width + 5, y_pos + baseline + 5), 
                (0, 0, 0), 
                -1
            )
            
            # Draw text
            cv.putText(
                frame, 
                text, 
                (text_x, y_pos), 
                font, 
                font_scale, 
                current_text_colour, 
                thickness
            )


def _add_body_part_legend(
    frame: np.ndarray, 
    body_part_colours: Dict[str, Tuple[int, int, int]],
    all_body_parts: Optional[Dict]
) -> None:
    """
    Add a legend showing body part colours on the right side of the frame.
    
    Parameters:
    -----------
    frame : np.ndarray
        Frame to add legend to
    body_part_colours : Dict[str, Tuple[int, int, int]]
        Dictionary of body part colours
    all_body_parts : Optional[Dict]
        Dictionary of detected body parts (to show only detected parts)
    """
    if not all_body_parts:
        return
    
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    
    # Position legend on right side
    frame_height, frame_width = frame.shape[:2]
    legend_x = frame_width - 200
    legend_y_start = 100
    legend_y_spacing = 25
    
    # Draw legend title
    cv.putText(
        frame, 
        "Body Parts", 
        (legend_x, legend_y_start - 10), 
        font, 
        font_scale + 0.1, 
        (255, 255, 255), 
        thickness + 1
    )
    
    # Draw each body part in legend (only if detected)
    for i, (body_part, colour) in enumerate(sorted(body_part_colours.items())):
        if body_part in all_body_parts:
            y_pos = legend_y_start + (i * legend_y_spacing)
            
            # Draw colour circle
            cv.circle(frame, (legend_x + 10, y_pos), 6, colour, -1)
            
            # Draw body part name
            cv.putText(
                frame, 
                body_part, 
                (legend_x + 25, y_pos + 5), 
                font, 
                font_scale, 
                colour, 
                thickness
            )


def _save_processing_summary(
    processed_trials: List[Dict], 
    skipped_trials: List[Dict], 
    processing_errors: List[Dict],
    output_path: Path,
    sort_by: str
) -> None:
    """
    Save a summary of processed trials including sorting method and angle correction statistics.
    
    Parameters:
    -----------
    processed_trials : List[Dict]
        List of processed trial information
    skipped_trials : List[Dict]
        List of skipped trial information
    processing_errors : List[Dict]
        List of processing error information
    output_path : Path
        Output directory path
    sort_by : str
        Sorting method used
    """
    summary_file = output_path / "processing_summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("DeepLabCut Performance Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        
        # Sorting information
        f.write(f"Images sorted by: {sort_by}\n")
        if processed_trials:
            if sort_by == 'cue_angle':
                min_val = min(t['cue_angle'] for t in processed_trials)
                max_val = max(t['cue_angle'] for t in processed_trials)
                f.write(f"Cue angle range: {min_val:.1f}° to {max_val:.1f}°\n\n")
            else:
                min_val = min(t['head_bearing'] for t in processed_trials)
                max_val = max(t['head_bearing'] for t in processed_trials)
                f.write(f"Head bearing range: {min_val:.1f}° to {max_val:.1f}°\n\n")
        
        total_trials = len(processed_trials) + len(skipped_trials) + len(processing_errors)
        f.write(f"Total trials: {total_trials}\n")
        f.write(f"Successfully processed: {len(processed_trials)}\n")
        f.write(f"Skipped during validation: {len(skipped_trials)}\n")
        f.write(f"Errors during processing: {len(processing_errors)}\n")
        f.write(f"Overall success rate: {len(processed_trials)/total_trials*100:.1f}%\n\n")
        
        f.write("Successfully processed trials (in sorted order):\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Sorted Idx':>10} {'Trial':>6} {'Cue Angle':>10} {'Bearing':>10} {'Filename'}\n")
        f.write("-" * 50 + "\n")
        
        for trial_info in processed_trials:
            f.write(
                f"{trial_info['sorted_idx']:10d} "
                f"{trial_info['trial_idx']:6d} "
                f"{trial_info['cue_angle']:10.1f}° "
                f"{trial_info['head_bearing']:10.1f}° "
                f"{trial_info['filename']}\n"
            )
        
        if skipped_trials:
            f.write(f"\nSkipped trials during validation:\n")
            f.write("-" * 30 + "\n")
            for skipped in skipped_trials:
                f.write(f"Trial {skipped['trial_idx']:3d}: {skipped['reason']}\n")
                
        if processing_errors:
            f.write(f"\nProcessing errors:\n")
            f.write("-" * 30 + "\n")
            for error in processing_errors:
                f.write(
                    f"Trial {error['trial_idx']:3d} "
                    f"(sorted position {error['sorted_idx']}): "
                    f"{error['reason']}\n"
                )


# Example usage
if __name__ == "__main__":
    # This example shows how to use the function
    # Replace with your actual session loading code
    
    from hex_behav_analysis.utils.Cohort_folder import Cohort_folder
    from hex_behav_analysis.utils.Session_nwb import Session
    
    model = "DLC_Resnet50_LMDCMar11shuffle1_snapshot_091"
    cohort_dir = r"/cephfs2/srogers/Behaviour/Pitx2_Chemogenetics/Experiment"
    
    cohort = Cohort_folder(cohort_dir, multi=True, OEAB_legacy=False, dlc_model_name=model, use_existing_cohort_info=True)
    test_dir = cohort.get_session("250521_133747_mtao106-3b")
    session = Session(test_dir, dlc_model_name=model, recalculate=True)
    
    # Example: Sort by cue angle
    evaluate_deeplabcut_performance(
        session=session,
        output_directory="/cephfs2/srogers/Behaviour/Pitx2_Chemogenetics/111_dlc_evaluation",
        likelihood_threshold=0,
        sort_by='cue_angle',  # or 'head_bearing'
        frame_offset=-3
    )
    
    print("\nDeepLabCut Performance Evaluation Script")
    print("Usage: evaluate_deeplabcut_performance(session, output_dir, likelihood_threshold, sort_by)")
    print("sort_by options: 'cue_angle' or 'head_bearing'")