import cv2
import time
import logging
import queue
import threading
import os
from config import Config

class CameraManager:
    """Manages camera/video capture and frame streaming with proper timing"""
    
    def __init__(self):
        self.config = Config()
        self.cap = None
        self.running = False
        self.frame_queue = queue.Queue(maxsize=self.config.FRAME_QUEUE_SIZE)
        
        # Frame rate control
        self.fps = 0
        self.frame_count = 0
        self.last_fps_update = time.time()
        self.target_fps = self.config.TARGET_FPS
        self.frame_interval = 1.0 / self.target_fps
        self.last_frame_time = time.time()
        
        # Video properties
        self.video_fps = 30  # Default, will be updated
        self.is_video_file = False
    
    def open_camera(self):
        """Initialize camera or load video file with proper timing"""
        # First try video file
        if os.path.exists(self.config.VIDEO_PATH):
            self.cap = cv2.VideoCapture(self.config.VIDEO_PATH)
            if self.cap.isOpened():
                self.is_video_file = True
                self.video_fps = self.cap.get(cv2.CAP_PROP_FPS)
                if self.video_fps <= 0:
                    self.video_fps = 30  # Default if cannot read FPS
                
                # Use video's native FPS for smooth playback
                self.target_fps = min(self.video_fps, self.config.TARGET_FPS)
                self.frame_interval = 1.0 / self.target_fps
                
                logging.info(f"🎞️ Loaded video: {self.config.VIDEO_PATH} (Native FPS: {self.video_fps}, Playback: {self.target_fps} FPS)")
                return True
        
        # If video file not found, try camera sources
        logging.warning("⚠️ Video file not found. Trying camera sources...")
        
        for backend in self.config.CAMERA_BACKENDS:
            for i in range(3):
                try:
                    self.cap = cv2.VideoCapture(i, backend)
                    if self.cap.isOpened():
                        # Camera settings
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.CAMERA_WIDTH)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.CAMERA_HEIGHT)
                        self.cap.set(cv2.CAP_PROP_FPS, self.config.CAMERA_FPS)
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        self.is_video_file = False
                        self.target_fps = self.config.TARGET_FPS
                        self.frame_interval = 1.0 / self.target_fps
                        
                        logging.info(f"🎥 Camera opened (Index: {i}), Target: {self.target_fps} FPS")
                        return True
                except Exception:
                    continue
        
        logging.error("❌ All camera and video sources failed.")
        return False
    
    def start_capture(self):
        """Start frame capture with proper timing control"""
        if not self.open_camera():
            return False
        
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        return True
    
    def _capture_loop(self):
        """Main capture loop with frame rate control"""
        last_frame_time = time.time()
        
        while self.running and self.cap and self.cap.isOpened():
            try:
                loop_start = time.time()
                
                success, frame = self.cap.read()
                if not success:
                    # Handle video end
                    if self.is_video_file:
                        if (self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.cap.get(cv2.CAP_PROP_FRAME_COUNT)):
                            logging.info("🔚 Video ended - restarting")
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                    time.sleep(0.01)
                    continue
                
                # Update FPS calculation
                self.frame_count += 1
                current_time = time.time()
                if current_time - self.last_fps_update >= 1.0:
                    self.fps = self.frame_count / (current_time - self.last_fps_update)
                    self.frame_count = 0
                    self.last_fps_update = current_time
                
                # Add frame to queue (non-blocking)
                if not self.frame_queue.full():
                    try:
                        self.frame_queue.put(frame.copy(), timeout=0.001)
                    except queue.Full:
                        pass  # Skip if queue full
                
                # CONTROL FRAME RATE - CRITICAL FOR SMOOTH PLAYBACK
                processing_time = time.time() - loop_start
                sleep_time = self.frame_interval - processing_time
                
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    # If we're running behind, log occasionally
                    if processing_time > self.frame_interval * 1.5:
                        logging.debug(f"⚠️ Capture lag: {processing_time:.3f}s")
                
                last_frame_time = time.time()
                
            except Exception as e:
                logging.error(f"Capture error: {e}")
                time.sleep(0.01)
    
    def get_frame(self):
        """Get frame from queue with timing control"""
        try:
            frame = self.frame_queue.get_nowait()
            return frame
        except queue.Empty:
            return None
    
    def get_fps(self):
        """Get current FPS"""
        return self.fps
    
    def get_target_fps(self):
        """Get target FPS"""
        return self.target_fps
    
    def stop_capture(self):
        """Stop frame capture"""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear queue
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        logging.info("🛑 Camera capture stopped")