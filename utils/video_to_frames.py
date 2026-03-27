import cv2
import os
import sys
from pathlib import Path

def extract_frames(video_path, output_dir, frame_interval=5):
    """Extract frames from video at specified intervals."""
    try:
        video = cv2.VideoCapture(str(video_path))
        if not video.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")
        
        frame_count = 0
        saved_count = 0
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            
            if frame_count % frame_interval == 0:
                output_path = os.path.join(output_dir, f'frame_{saved_count:04d}.jpg')
                cv2.imwrite(output_path, frame)
                saved_count += 1
                print(f"Saved frame {saved_count}")
            
            frame_count += 1
        
        video.release()
        return saved_count
        
    except Exception as e:
        print(f"Error in extract_frames: {str(e)}")
        raise

def process_videos():
    """Process a video and extract frames."""
    if len(sys.argv) < 3:
        print("Usage: python video_to_frames.py <video_path> <type>")
        print("type: 'real' or 'fake'")
        sys.exit(1)
    
    video_path = sys.argv[1]
    video_type = sys.argv[2]
    
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        sys.exit(1)
    
    output_dir = os.path.join('dataset', video_type)
    frames = extract_frames(video_path, output_dir)
    print(f"Successfully extracted {frames} frames from {video_path}")

if __name__ == "__main__":
    process_videos()
    