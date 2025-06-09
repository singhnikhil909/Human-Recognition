# Person Detection and Tracking Tool

This tool detects and tracks people in a video, recording their presence timestamps and optionally creating a labeled video with bounding boxes.

## Features

- Detects multiple people in video frames
- Tracks individual people across frames
- Records timestamps of when each person appears in the video
- Generates JSON output with person presence information
- Optional: Creates a labeled video with bounding boxes around detected people

## Requirements

- Python 3.7 or higher
- OpenCV
- NumPy
- Ultralytics (YOLOv8)
- tqdm

## Installation

1. Clone this repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage
Run the script:
```bash
python app.py
```
It will run on the development server http://127.0.0.1:5000

The script will:
- Process the video and detect people
- Generate a JSON file (`person_tracks.json`) with timestamps
- Optionally create a labeled video (`output_video.mp4`) with bounding boxes
  

## Output Format

The JSON output will be in the following format:
```json
{
    "Person 1": [
        {
            "start": "00:00:00",
            "end": "00:01:00"
        },
        {
            "start": "00:01:40",
            "end": "00:03:20"
        }
    ],
    "Person 2": [
        {
            "start": "00:00:10",
            "end": "00:02:30"
        }
    ]
}
```

Each person is assigned a unique ID, and their presence in the video is recorded as a list of time segments.

## Notes

- The tool uses YOLOv8 for person detection
- A confidence threshold of 0.5 is used for detections
- Time segments are merged if the gap between appearances is less than 1 second
- The tracking algorithm uses a simple overlap-based approach to maintain person identity 
