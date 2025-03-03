#!/usr/bin/env python3
"""
script plays a video file in the terminal using high-density Braille rendering.
synchronized audio playback handled via ffplay. 
The video is scaled to preserve its original aspect ratio by default, or can be stretched to fill the terminal with the -f flag.

Usage: python3 terminal_mp4_player.py [options] /path/to/video.mp4

Flags: 
  -f, --fill          Fill terminal (stretch video to fit, ignoring aspect ratio).
  -g, --gamma FLOAT   Gamma correction value (default: 2.2). Controls brightness and color.
                      Higher values (2.5-3.0) increase brightness, lower values (1.5-1.8) darken.
  -i, --interpolate   Enable frame interpolation for smoother animation (default: enabled).
                      Creates intermediate frames for fluid motion.
  --no-interpolate    Disable frame interpolation. Use for lower CPU usage.
  -f, --fill          Fill terminal (stretch video to fit, ignoring aspect ratio).
                      Without this flag, video maintains correct aspect ratio.
  --no-vsync          Disable vertical sync (frame buffering). May increase performance
                      but might cause visual tearing.

Example: python3 terminal_mp4_player.py -f video.mp4

Press Ctrl+C to exit.
"""

import cv2
import numpy as np
import argparse
import os
import sys
import time
import subprocess
import signal
import shutil
import logging

# ANSI escape codes
CSI = "\x1b["
ALTERNATE_SCREEN_ON = CSI + "?1049h"
ALTERNATE_SCREEN_OFF = CSI + "?1049l"
HIDE_CURSOR = CSI + "?25l"
SHOW_CURSOR = CSI + "?25h"
CLEAR_SCREEN = CSI + "2J"
RESET_COLOR = "\x1b[0m"
MOVE_CURSOR_HOME = CSI + "H"
SAVE_CURSOR = CSI + "s"
RESTORE_CURSOR = CSI + "u"

# Global flags
exit_flag = False
debug_mode = False

def signal_handler(sig, frame):
    global exit_flag
    exit_flag = True

signal.signal(signal.SIGINT, signal_handler)

def get_terminal_size():
    size = shutil.get_terminal_size((80, 24))
    return size.columns, size.lines


# Global gamma value for color correction
GAMMA_VALUE = 2.2

def render_frame_braille(frame):
    """
    Convert an RGB frame (numpy array) into a string using Unicode Braille characters.
    Each Braille character maps a 2x4 block. A pixel is 'on' if its luminance > threshold.
    The average color of the 'on' pixels is applied via ANSI truecolor escapes with gamma correction.
    """
    # Helper function for conditional debug printing
    def debug_print(message):
        if debug_mode:
            logging.debug(message)
    try:
        height, width, _ = frame.shape
        # Only do expensive NaN checks in debug mode
        if debug_mode:
            if np.isnan(frame).any() or np.isinf(frame).any():
                logging.debug("Warning: Frame contains NaN or Inf values, replacing with zeros")
                frame = np.nan_to_num(frame, nan=0, posinf=255, neginf=0)
            
        # Ensure width is even for 2-column braille patterns
        if width % 2 != 0:
            frame = np.pad(frame, ((0, 0), (0, 1), (0, 0)), mode='edge')
            width += 1
        # Ensure height is divisible by 4 for 4-row braille patterns
        if height % 4 != 0:
            pad_h = 4 - (height % 4)
            frame = np.pad(frame, ((0, pad_h), (0, 0), (0, 0)), mode='edge')
            height += pad_h
            
        # Make sure frame data is uint8 and within bounds
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    
        # Bit patterns for each position in the 2x4 grid of a braille character
        # Using Python integer literals instead of numpy array to avoid any potential type issues
        dot_bit = [
            [0x01, 0x08],  # Top row (dots 1, 4)
            [0x02, 0x10],  # Second row (dots 2, 5)
            [0x04, 0x20],  # Third row (dots 3, 6)
            [0x40, 0x80]   # Bottom row (dots 7, 8)
        ]
    
        out_lines = []
        # Process the image in 2x4 blocks (the size of one braille character)
        for y in range(0, height, 4):
            line = []
            for x in range(0, width, 2):
                try:
                    # Extract the 2x4 block, ensuring we don't go out of bounds
                    cell = 0  # Initialize the braille cell bit pattern
                    y_end = min(y + 4, height)
                    x_end = min(x + 2, width)
                    block = frame[y:y_end, x:x_end, :]
                    
                    # If we have a partial block (at edges), pad it
                    if block.shape[0] < 4 or block.shape[1] < 2:
                        padded_block = np.zeros((4, 2, 3), dtype=np.uint8)
                        padded_block[:block.shape[0], :block.shape[1], :] = block
                        block = padded_block
                    
                    # Calculate luminance (brightness) for each pixel using the Rec. 709 standard
                    # This provides a more perceptually accurate luminance calculation
                    lum = 0.2126 * block[:, :, 0] + 0.7152 * block[:, :, 1] + 0.0722 * block[:, :, 2]
                    
                    # Use a dynamic threshold based on the average luminance of the block
                    # but ensure that darker areas still have some dots turned on
                    avg_lum = np.mean(lum)
                    
                    # Calculate a smoother dynamic threshold that's lower for darker areas
                    # and higher for brighter areas to maintain edge contrast
                    base_threshold = 30  # Lower base threshold for smoother appearance
                    max_threshold = 120   # Lower max threshold for smoother appearance
                    
                    # Use a smooth sigmoid-like function to transition between thresholds
                    # This creates a more gradual change in threshold rather than sharp cutoffs
                    if avg_lum < 20:
                        # Very dark areas get a very low threshold
                        threshold = max(5, avg_lum / 4)  # Even lower threshold for dark areas
                    elif avg_lum < 50:
                        # Dark to mid areas get a smoothly increasing threshold
                        # Linear interpolation between low and base threshold
                        t = (avg_lum - 20) / 30.0  # 0 to 1 as avg_lum goes from 20 to 50
                        threshold = (1 - t) * (avg_lum / 4) + t * base_threshold
                    else:
                        # Mid to bright areas get a threshold between base and max
                        # Use a smooth curve rather than linear scaling
                        t = min(1.0, (avg_lum - 50) / 150.0)  # 0 to 1 as avg_lum goes from 50 to 200
                        # Smooth curve: t^2 * (3-2t) gives a smooth S-curve from 0 to 1
                        smooth_t = t * t * (3 - 2 * t)
                        threshold = base_threshold + (max_threshold - base_threshold) * smooth_t
                    
                    # Determine which dots should be on (threshold based)
                    dots = (lum > threshold)
                    
                    # Ensure at least one dot is on in each block to avoid empty areas
                    if not np.any(dots):
                        # If no dots are on, turn on the brightest one
                        brightest_pos = np.unravel_index(np.argmax(lum), lum.shape)
                        dots[brightest_pos] = True
                    
                    # Calculate the braille cell value based on which dots are on
                    # Start with cell as a pure Python integer
                    cell = 0
                    for i in range(4):
                        for j in range(2):
                            if i < dots.shape[0] and j < dots.shape[1] and dots[i, j]:
                                # Use pure Python integer operations
                                cell = cell | dot_bit[i][j]  # Using Python's bitwise OR operator
                                
                    # Calculate the average color based on which dots are on
                    # If no dots are on (which shouldn't happen due to our fix above),
                    # fall back to using the entire block
                    if np.count_nonzero(dots) > 0:
                        # Extract colors of the "on" pixels
                        on_pixels = block[dots]
                        # Calculate average color
                        avg_color = np.mean(on_pixels, axis=0)
                    else:
                        # Fallback - use average of entire block
                        avg_color = np.mean(block, axis=(0, 1))
                    
                    # Apply enhanced gamma correction for more vibrant colors
                    gamma_factor = 1.0/GAMMA_VALUE
                    avg_color = np.power(avg_color / 255.0, gamma_factor) * 255.0
                    
                    # Enhance color saturation by scaling distance from gray for more vibrant colors
                    gray_value = np.mean(avg_color)
                    saturation_factor = 1.3  # Increase saturation by 30% for more vibrant colors
                    for i in range(3):
                        # Scale the distance from gray
                        avg_color[i] = gray_value + saturation_factor * (avg_color[i] - gray_value)
                    
                    # Apply stronger contrast enhancement for better definition
                    contrast_factor = 1.2  # Increase contrast by 20% for better definition
                    avg_color = ((avg_color / 255.0 - 0.5) * contrast_factor + 0.5) * 255.0
                    
                    # Ensure that if any channel has a non-zero value, the color is never too dark
                    if np.any(avg_color > 0):
                        # Find the maximum channel value
                        max_channel = np.max(avg_color)
                        # If the max channel is below our minimum threshold, scale all channels up
                        min_threshold = 20  # Minimum brightness threshold
                        if max_channel < min_threshold and max_channel > 0:
                            # Scale all channels proportionally
                            scale = min_threshold / max_channel
                            avg_color = np.minimum(avg_color * scale, 255.0)
                        
                    # Generate ANSI true color escape sequence with explicit integer conversion
                    # Ensure each color component is a valid integer between 0-255
                    r = min(255, max(0, int(avg_color[0])))
                    g = min(255, max(0, int(avg_color[1])))
                    b = min(255, max(0, int(avg_color[2])))
                    
                    # Construct the escape sequence with guaranteed valid integers
                    fg = f"{CSI}38;2;{r};{g};{b}m"
                    
                    # Create the Unicode braille character (ensure cell is a valid integer)
                    # The Unicode braille patterns start at 0x2800 (10240 decimal)
                    # cell must be between 0 and 255 to be valid
                    cell = min(255, max(0, int(cell)))
                    braille_char = chr(0x2800 + cell)
                    # Combine color and character
                    cell_str = fg + braille_char + RESET_COLOR
                except Exception as e:
                    debug_print(f"Error processing block at ({x},{y}): {e}")
                    # Use a space character if there's an error
                    cell_str = " "
                    
                line.append(cell_str)
            out_lines.append(''.join(line))
        # Join all lines with newlines
        return "\n".join(out_lines)
    except Exception as e:
        logging.error(f"Error in render_frame_braille: {e}")
        # Return a fallback pattern in case of errors
        return "\n".join(["ERROR: Could not render frame"] * 10)


# Define VSYNC constants for frame rendering
VSYNC_ENABLED = True
CURSOR_TO_TOP = CSI + "1;1H"

def debug_print(frame):
    """Render the video frame in the terminal using ANSI true color blocks with background color for maximum visibility."""
    for row in frame:
        line = ""
        for pixel in row:
            # Assuming pixel is a tuple (R, G, B)
            r, g, b = pixel
            
            # Ensure that if any channel has a non-zero value, none of them are completely black
            if (r > 0 or g > 0 or b > 0) and (r < 10 or g < 10 or b < 10):
                # If any channel is non-zero but very dark, boost all channels slightly
                # to ensure visibility while preserving relative color ratios
                min_val = 10  # Minimum value to ensure visibility
                max_channel = max(r, g, b)
                if max_channel > 0:  # Avoid division by zero
                    # Scale all channels proportionally so the max is at least min_val
                    scale = max(min_val / max_channel, 1.0)
                    r = min(255, int(r * scale))
                    g = min(255, int(g * scale))
                    b = min(255, int(b * scale))
            
            # Use a space character with background color instead of foreground color
            # This ensures the entire cell is filled with color
            line += f"\033[48;2;{r};{g};{b}m \033[0m"
        print(line)


def play_video(video_path, debug=False, gamma=2.2, interpolate=True, fill_mode=False):
    global GAMMA_VALUE
    GAMMA_VALUE = gamma
    global debug_mode
    debug_mode = debug
    
    # Configure logging - only set up once
    if debug_mode:
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
    print(f"Starting playback of: {video_path}")
    if debug_mode:
        logging.debug(f"Debug mode: {debug_mode}")
    if not os.path.isfile(video_path):
        print(f"Error: Video file not found: {video_path}", flush=True)
        sys.exit(1)

    # Check for ffplay for audio synchronization
    try:
        ffplay_check = subprocess.run(['which', 'ffplay'],
                                      stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE,
                                      text=True,
                                      check=False)
        if ffplay_check.returncode != 0:
            print("Warning: ffplay not found. Audio playback will be disabled.", flush=True)
            has_ffplay = False
        else:
            has_ffplay = True
            print(f"Found ffplay at: {ffplay_check.stdout.strip()}", flush=True)
    except Exception as e:
        print(f"Warning: Could not check for ffplay: {e}", flush=True)
        has_ffplay = False

    # We'll start audio playback after video is ready to render
    audio_process = None
    audio_command = None
    if has_ffplay:
        # Prepare the audio command but don't start it yet
        audio_command = [
            'ffplay', '-nodisp', '-autoexit', '-loglevel', 'quiet', video_path
        ]
        print("Audio playback prepared, will start with video", flush=True)
    else:
        print("Skipping audio playback", flush=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: OpenCV cannot open video file {video_path}", flush=True)
        sys.exit(1)
    else:
        print("Video file opened successfully", flush=True)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    print(f"Video properties: {width}x{height}, {fps} FPS, {frame_count} frames", flush=True)
    if fps <= 0:
        print("Warning: Invalid FPS; defaulting to 24", flush=True)
        fps = 24
    frame_duration = 1.0 / fps

    # Determine target resolution based on terminal size (each braille char = 2x4 pixels)
    term_cols, term_rows = get_terminal_size()
    # Subtract 1 from rows to prevent potential scrolling issues
    term_rows = max(1, term_rows - 1)
    target_width = term_cols * 2
    target_height = term_rows * 4
    # Ensure dimensions are valid (at least 8x8)
    target_width = max(8, target_width)
    target_height = max(8, target_height)
    print(f"Target resolution: {target_width}x{target_height}")

    # Set up terminal display: alternate screen and hide cursor
    # Also set the terminal to raw mode to reduce input buffering delay
    sys.stdout.write(ALTERNATE_SCREEN_ON + HIDE_CURSOR)
    sys.stdout.flush()
    
    # Try to optimize terminal output for performance
    try:
        # Disable terminal echo and line buffering if possible
        os.system('stty -echo -icanon min 1 time 0')
    except Exception:
        pass

    # Brief pause to let video capture initialize
    time.sleep(0.5)

    # Verify first frame can be read
    ret, frame = cap.read()
    if not ret or frame is None:
        print("Error: Failed to read first frame")
        sys.exit(1)
    # Reset to beginning of video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    print("Entering playback loop...", flush=True)
    frame_index = 0
    frame_read_timeout = 5.0  # seconds per frame
    
    # We'll set the video start time when we actually start rendering
    video_start_time = None

    try:
        while not exit_flag:
            # Skip playback loop iteration debug output
            retries = 0
            ret = False
            frame = None
            if frame_index > 0:
                pass
            else:
                pass
            
            frame_read_start = time.time()
            while not ret and (time.time() - frame_read_start) < frame_read_timeout and not exit_flag:
                try:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        retries += 1
                        time.sleep(min(0.1, 0.5 / (retries + 1)))
                        if retries % 10 == 0:
                            elapsed_local = time.time() - frame_read_start
                            print(f"Still reading, attempt {retries}, elapsed: {elapsed_local:.1f}s", flush=True)
                            cap.grab()
                except Exception as e:
                    print(f"Exception during frame read: {e}", flush=True)
                    time.sleep(0.1)
                    retries += 1
            if not ret or frame is None:
                elapsed_local = time.time() - frame_read_start
                print(f"Error: Couldn't read frame {frame_index+1} after {elapsed_local:.1f}s ({retries} attempts)", flush=True)
                if frame_index > 0:
                    try:
                        print("Recovery: Resetting video position...", flush=True)
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            print("Recovery successful", flush=True)
                        else:
                            print("Recovery failed", flush=True)
                            break
                    except Exception as e:
                        print(f"Recovery exception: {e}", flush=True)
                        break
                else:
                    print("Failed to read first frame. Exiting.", flush=True)
                    break
            else:
                if frame_index == 0:
                    # First frame read successfully, no need to print details
                    pass

            
            # Check if dimensions are reversed (height, width instead of width, height)
            height, width = frame.shape[:2]

            
            # Resize frame to fit terminal dimensions
            # Ensure dimensions are in the correct order (width, height)
            # OpenCV resize takes (width, height) but frame.shape is (height, width, channels)

            try:
                # Make sure target dimensions are valid
                if target_width < 2 or target_height < 4:
                    print(f"Warning: Target dimensions too small, using minimum size", flush=True)
                    target_width = max(2, target_width)
                    target_height = max(4, target_height)
                
                # Ensure dimensions match braille character grid (multiples of 2x4)
                target_width = target_width - (target_width % 2) if target_width % 2 != 0 else target_width
                target_height = target_height - (target_height % 4) if target_height % 4 != 0 else target_height
                
                if fill_mode:
                    # Original behavior: stretch to fill the terminal (ignore aspect ratio)
                    frame = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                else:
                    # Preserve aspect ratio
                    original_height, original_width = frame.shape[:2]
                    aspect_ratio = original_width / original_height
                    
                    # Calculate dimensions that preserve aspect ratio
                    if target_width / target_height > aspect_ratio:
                        # Terminal is wider than video aspect ratio
                        new_width = int(target_height * aspect_ratio)
                        new_height = target_height
                    else:
                        # Terminal is taller than video aspect ratio
                        new_width = target_width
                        new_height = int(target_width / aspect_ratio)
                    
                    # Ensure dimensions are even/multiples of 4 for braille characters
                    new_width = new_width - (new_width % 2) if new_width % 2 != 0 else new_width
                    new_height = new_height - (new_height % 4) if new_height % 4 != 0 else new_height
                    
                    # Resize the frame to the new dimensions
                    resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
                    
                    # Create a black frame of the target size
                    frame = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                    
                    # Calculate position to center the video
                    y_offset = (target_height - new_height) // 2
                    x_offset = (target_width - new_width) // 2
                    
                    # Place the resized frame in the center of the black frame
                    frame[y_offset:y_offset+new_height, x_offset:x_offset+new_width] = resized_frame

            except Exception as e:
                logging.error(f"Resize error: {e}. Using original frame.")
            
            # Convert from BGR to RGB color space
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Apply a gentler sharpening filter for smoother appearance
            kernel = np.array([[-0.5, -0.5, -0.5],
                               [-0.5, 5.0, -0.5],
                               [-0.5, -0.5, -0.5]])
            frame = cv2.filter2D(frame, -1, kernel)
            
            # Enhance contrast
            # Convert to LAB color space (L = lightness, A/B = color channels)
            lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
            # Split channels
            l, a, b = cv2.split(lab)
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to lightness channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            # Merge channels back
            merged = cv2.merge((cl, a, b))
            # Convert back to RGB
            frame = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
            
            # Ensure frame data is uint8 and within bounds
            frame = np.clip(frame, 0, 255).astype(np.uint8)
            # Render frame using braille characters
            rendered = render_frame_braille(frame)
            
            # Instead of clearing the screen (which causes flickering),
            # save cursor position, move to home, render, and restore cursor
            # This reduces the visible flickering between frames
            # Start audio playback and set timing - moved out of rendering section
            if frame_index == 0:                
                # Start audio playback at the exact moment we display the first frame
                # This ensures audio and video start together
                if audio_command and not audio_process:
                    try:
                        print("Starting audio playback synchronized with video...", flush=True)
                        audio_process = subprocess.Popen(
                            audio_command, 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL
                        )
                        # Set the video start time when we actually start rendering and audio
                        video_start_time = time.time()
                        print("Audio playback started successfully", flush=True)
                    except Exception as e:
                        print(f"Warning: Audio playback failed: {e}", flush=True)
                        # If audio fails, still set start time for video timing
                        video_start_time = time.time()
                else:
                    # No audio, just set the start time
                    video_start_time = time.time()
                
            # Buffer and write the full frame at once to prevent tearing
            if VSYNC_ENABLED:
                # For frame-by-frame rendering, write the cursor position and full frame in a single write operation
                # This minimizes the chance of tearing since the terminal will process it as one update
                if frame_index == 0:
                    full_frame_buffer = CLEAR_SCREEN + MOVE_CURSOR_HOME + rendered
                else:
                    full_frame_buffer = MOVE_CURSOR_HOME + rendered
                # Write the entire buffer as a single operation
                sys.stdout.write(full_frame_buffer)
            else:
                # Original line-by-line approach
                if frame_index == 0:
                    sys.stdout.write(CLEAR_SCREEN + MOVE_CURSOR_HOME)
                else:
                    sys.stdout.write(MOVE_CURSOR_HOME)
                sys.stdout.write(rendered)
                
            # Ensure output is displayed immediately
            sys.stdout.flush()
            
            # Update frame index
            frame_index += 1
            
            # Get current frame's timestamp in seconds from the video
            frame_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            
            # Calculate elapsed time since video playback started
            elapsed_time = time.time() - video_start_time
            
            # Calculate the ideal frame position based on elapsed time and the original video FPS
            # This ensures we maintain proper sync with audio
            ideal_frame = int(elapsed_time * fps)
            
            # For ultra-smooth animation, we'll use advanced frame interpolation
            # This creates intermediate frames between source frames for smoother motion
            # while still maintaining the original timing and sync
            
            # Calculate how many frames ahead or behind we are compared to ideal position
            frame_diff = frame_index - ideal_frame
            
            # Precise synchronization approach that maintains original video timing
            if frame_diff > 2:  # Video is ahead of audio by more than 2 frames
                # We're ahead, add a delay to let audio catch up
                # The delay is proportional to how far ahead we are
                delay_time = min(0.05, 0.01 * frame_diff)  # Cap at 50ms
                time.sleep(delay_time)
            elif frame_diff < -2:  # Video is behind audio by more than 2 frames
                # We're behind, skip frames to catch up with audio
                skip_frames = min(10, -frame_diff)  # Allow skipping up to 10 frames if needed
                if skip_frames > 0:
                    # Use efficient frame skipping with cap.set
                    next_frame_pos = frame_index + skip_frames
                    cap.set(cv2.CAP_PROP_POS_FRAMES, next_frame_pos)
                    frame_index = next_frame_pos
            else:
                # We're in sync (within 2 frames)
                # Calculate the exact time we should wait to maintain proper sync
                target_time = video_start_time + (frame_index / fps)
                current_time = time.time()
                wait_time = max(0, target_time - current_time)
                
                # If interpolation is enabled and we have enough extra time, create and display interpolated frames
                # This creates much smoother animation without affecting sync
                if interpolate and wait_time > 0.033:  # If we have more than 33ms to wait (enough time for interpolation)
                    try:
                        # Read the next frame without advancing the frame counter
                        success, next_frame = cap.read()
                        if success:
                            # Store the position so we can return to it
                            current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
                            # Go back one frame so we don't skip it in the main loop
                            cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos - 1)
                            
                            # Determine how many interpolated frames we can show based on wait time
                            # More wait time = more intermediate frames = smoother animation
                            # Allow up to 7 interpolated frames for ultra-smooth motion
                            num_interpolated_frames = min(7, int(wait_time / 0.008))
                            frame_wait_time = wait_time / (num_interpolated_frames + 1)
                            
                            # Create and display multiple interpolated frames
                            for i in range(1, num_interpolated_frames + 1):
                                # Calculate blend factor (position between frames)
                                alpha = i / (num_interpolated_frames + 1)
                                
                                # Create interpolated frame using weighted average
                                interpolated_frame = cv2.addWeighted(frame, 1-alpha, next_frame, alpha, 0)
                                
                                # Apply advanced motion blur and frame enhancement techniques
                                if i > 1:
                                    # Apply temporal motion blur by blending with previous interpolated frame
                                    prev_weight = 0.15  # Lighter motion blur for sharper image
                                    interpolated_frame = cv2.addWeighted(prev_interpolated, prev_weight, 
                                                                       interpolated_frame, 1-prev_weight, 0)
                                    
                                # Apply subtle edge enhancement for sharper details
                                # This makes the braille dots appear more defined
                                kernel = np.array([[-0.5, -0.5, -0.5], 
                                                  [-0.5,  5.0, -0.5], 
                                                  [-0.5, -0.5, -0.5]]) * 0.15
                                enhanced = cv2.filter2D(interpolated_frame, -1, kernel)
                                # Blend the enhanced image with the original for a subtle effect
                                interpolated_frame = cv2.addWeighted(interpolated_frame, 0.85, enhanced, 0.15, 0)
                                
                                # Store for potential motion blur in next frame
                                prev_interpolated = interpolated_frame.copy()
                                
                                # Render and display the interpolated frame
                                interpolated_rendered = render_frame_braille(interpolated_frame)
                                
                                # Use the same frame buffering technique for interpolated frames
                                if VSYNC_ENABLED:
                                    # Write the entire frame at once
                                    full_buffer = CURSOR_TO_TOP + interpolated_rendered
                                    sys.stdout.write(full_buffer)
                                else:
                                    # Original approach
                                    sys.stdout.write(CURSOR_TO_TOP)
                                    sys.stdout.write(interpolated_rendered)
                                
                                sys.stdout.flush()
                                
                                # Sleep until it's time for the next interpolated frame
                                time.sleep(frame_wait_time)
                    except Exception as e:
                        # If interpolation fails, fall back to normal waiting
                        logging.debug(f"Interpolation error: {e}")
                        time.sleep(wait_time * 0.7)
                elif wait_time > 0.001:  # If we have a smaller amount of time
                    # Use a portion of the wait time
                    time.sleep(wait_time * 0.7)
        print("Playback loop ended.", flush=True)
    except Exception as e:
        print(f"Error during playback: {e}", flush=True)
    finally:
        cap.release()
        if audio_process:
            audio_process.terminate()
        # Restore terminal settings
        try:
            os.system('stty echo icanon')
        except Exception:
            pass
        sys.stdout.write(RESET_COLOR + SHOW_CURSOR + ALTERNATE_SCREEN_OFF)
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Play a mp4 in a true color terminal with braille ascii rendering.")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug output mode")
    parser.add_argument("-g", "--gamma", type=float, default=2.2, help="Gamma correction value (default: 2.2)")
    parser.add_argument("-i", "--interpolate", action="store_true", default=True, help="Enable frame interpolation for smoother animation (default: enabled)")
    parser.add_argument("--no-interpolate", action="store_false", dest="interpolate", help="Disable frame interpolation")
    parser.add_argument("-f", "--fill", action="store_true", help="Fill terminal (stretch video to fit, ignoring aspect ratio)")
    parser.add_argument("--no-vsync", action="store_true", help="Disable vertical sync (frame buffering)")
    args = parser.parse_args()
    
    # Set the global VSYNC_ENABLED flag
    global VSYNC_ENABLED
    VSYNC_ENABLED = not args.no_vsync
    
    play_video(args.video, debug=args.debug, gamma=args.gamma, interpolate=args.interpolate, fill_mode=args.fill)


if __name__ == '__main__':
    main()
