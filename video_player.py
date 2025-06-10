import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import os

class VideoPlayer:
    def __init__(self, video_path):
        self.root = tk.Tk()
        self.root.title("Processed Video Player")
        
        # Create main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10)
        
        # Create video label
        self.video_label = ttk.Label(self.main_frame)
        self.video_label.pack()
        
        # Create control buttons
        self.control_frame = ttk.Frame(self.main_frame)
        self.control_frame.pack(pady=5)
        
        self.play_button = ttk.Button(self.control_frame, text="Play", command=self.play_video)
        self.play_button.pack(side=tk.LEFT, padx=5)
        
        self.pause_button = ttk.Button(self.control_frame, text="Pause", command=self.pause_video)
        self.pause_button.pack(side=tk.LEFT, padx=5)
        
        # Video properties
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            messagebox.showerror("Error", f"Could not open video file: {video_path}")
            self.root.destroy()
            return
            
        self.is_playing = False
        
        # Get video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.delay = int(1000/self.fps)
        
        # Start the UI
        self.root.mainloop()
    
    def play_video(self):
        self.is_playing = True
        self.update_frame()
    
    def pause_video(self):
        self.is_playing = False
    
    def update_frame(self):
        if self.is_playing:
            ret, frame = self.cap.read()
            if ret:
                # Convert frame to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Resize frame to fit window
                height, width = frame.shape[:2]
                max_height = 600
                if height > max_height:
                    ratio = max_height / height
                    width = int(width * ratio)
                    height = max_height
                    frame = cv2.resize(frame, (width, height))
                
                # Convert to PhotoImage
                image = Image.fromarray(frame)
                photo = ImageTk.PhotoImage(image=image)
                
                # Update label
                self.video_label.configure(image=photo)
                self.video_label.image = photo
                
                # Schedule next frame
                self.root.after(self.delay, self.update_frame)
            else:
                # Reset video to beginning
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.update_frame()

def main():
    # Get the processed video path from the output directory
    output_dir = "output"
    
    if not os.path.exists(output_dir):
        messagebox.showerror("Error", f"Output directory '{output_dir}' does not exist!")
        return
        
    video_files = [f for f in os.listdir(output_dir) if f.endswith('_processed.mp4')]
    
    if video_files:
        video_path = os.path.join(output_dir, video_files[0])
        print(f"Found processed video: {video_path}")
        player = VideoPlayer(video_path)
    else:
        messagebox.showerror("Error", "No processed video found in the output directory. Please run person_detector.py first.")
        print("No processed video found in the output directory.")

if __name__ == "__main__":
    main() 