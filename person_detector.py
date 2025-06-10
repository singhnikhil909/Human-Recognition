import cv2
import numpy as np
from ultralytics import YOLO
import json
from datetime import timedelta
from tqdm import tqdm
import os

class PersonDetector:
    def __init__(self, video_path, output_video_path=None):
        self.video_path = video_path
        # Create output directory if it doesn't exist
        self.output_dir = "output"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set default output path if none provided
        if output_video_path is None:
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            self.output_video_path = os.path.join(self.output_dir, f"{video_name}_processed.mp4")
        else:
            self.output_video_path = output_video_path
            
        self.model = YOLO('yolov8n.pt')  # Load YOLOv8 model
        self.person_tracks = {}  # Dictionary to store person tracks
        self.next_person_id = 1  # Counter for assigning person IDs
        
    def format_time(self, seconds):
        """Convert seconds to HH:MM:SS format"""
        return str(timedelta(seconds=int(seconds)))
    
    def process_video(self):
        # Open the video file
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError("Error: Could not open video file")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # Use H.264 codec
        writer = cv2.VideoWriter(
            self.output_video_path,
            fourcc,
            fps,
            (width, height)
        )
        
        if not writer.isOpened():
            raise ValueError("Error: Could not create output video file")
        
        # Process video frame by frame
        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Get current timestamp
                current_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
                
                # Run YOLOv8 inference
                results = self.model(frame, classes=[0])  # class 0 is person in COCO dataset
                
                # Process detections
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = float(box.conf[0])
                        
                        if confidence > 0.7:  # Increased confidence threshold to 0.7
                            # Check if this detection matches any existing track
                            matched = False
                            best_match_id = None
                            best_overlap = 0.5  # Minimum overlap threshold
                            
                            for person_id, track in self.person_tracks.items():
                                last_box = track['boxes'][-1]
                                overlap = self._calculate_overlap((x1, y1, x2, y2), last_box)
                                if overlap > best_overlap:
                                    best_overlap = overlap
                                    best_match_id = person_id
                                    matched = True
                            
                            if matched:
                                # Update the best matching track
                                self.person_tracks[best_match_id]['times'].append(current_time)
                                self.person_tracks[best_match_id]['boxes'].append((x1, y1, x2, y2))
                            else:
                                # Create new track
                                self.person_tracks[self.next_person_id] = {
                                    'times': [current_time],
                                    'boxes': [(x1, y1, x2, y2)]
                                }
                                self.next_person_id += 1
                            
                            # Draw bounding box
                            person_id = best_match_id if matched else self.next_person_id - 1
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"Person {person_id}", 
                                      (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
                                      (0, 255, 0), 2)
                
                # Write the frame
                writer.write(frame)
                pbar.update(1)
        
        # Release resources
        cap.release()
        writer.release()
        
        # Verify the output video was created
        if not os.path.exists(self.output_video_path):
            raise ValueError(f"Error: Output video was not created at {self.output_video_path}")
        
        return self._generate_json_output()
    
    def _calculate_overlap(self, box1, box2):
        """Calculate IoU (Intersection over Union) between two bounding boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = box1_area + box2_area - intersection
        return intersection / union if union > 0 else 0
    
    def _generate_json_output(self):
        """Generate JSON output with person tracks"""
        output = {}
        for person_id, track in self.person_tracks.items():
            # Merge consecutive time segments
            times = track['times']
            segments = []
            start_time = times[0]
            prev_time = times[0]
            
            for i in range(1, len(times)):
                if times[i] - prev_time > 1.0:  # Gap threshold of 1 second
                    segments.append({
                        "start": self.format_time(start_time),
                        "end": self.format_time(prev_time)
                    })
                    start_time = times[i]
                prev_time = times[i]
            
            # Add the last segment
            segments.append({
                "start": self.format_time(start_time),
                "end": self.format_time(prev_time)
            })
            
            output[f"Person {person_id}"] = segments
        
        return output

def main():
    # Example usage
    video_path = r"C:\Users\NikhilSingh\Downloads\face-demographics-walking-and-pause.mp4" # Replace with your video path
    
    try:
        detector = PersonDetector(video_path)
        results = detector.process_video()
        
        # Save results to JSON file in output directory
        output_json_path = os.path.join(detector.output_dir, "person_tracks.json")
        with open(output_json_path, "w") as f:
            json.dump(results, f, indent=4)
        
        print(f"Processing complete!")
        print(f"Results saved to: {output_json_path}")
        print(f"Processed video saved to: {detector.output_video_path}")
        
        # Return paths for UI to use
        return {
            "processed_video_path": detector.output_video_path,
            "json_path": output_json_path
        }
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return None

if __name__ == "__main__":
    main() 