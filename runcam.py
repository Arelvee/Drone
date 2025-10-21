import cv2
import threading
import time
from datetime import datetime

class RunCamWifiLink2:
    def __init__(self):
        self.cap = None
        self.is_connected = False
        self.current_frame = None
        self.thread = None
        self.running = False
        
    def connect(self, rtsp_url="rtsp://192.168.1.1/stream1"):
        """Connect to RunCam WifiLink 2"""
        try:
            self.cap = cv2.VideoCapture(rtsp_url)
            
            # Set buffer size to minimize latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if self.cap.isOpened():
                self.is_connected = True
                print(f"Connected to RunCam at {rtsp_url}")
                return True
            else:
                print("Failed to connect to RunCam")
                return False
                
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def start_stream(self):
        """Start streaming in a separate thread"""
        if not self.is_connected:
            print("Not connected to camera")
            return False
            
        self.running = True
        self.thread = threading.Thread(target=self._stream_worker)
        self.thread.daemon = True
        self.thread.start()
        return True
    
    def _stream_worker(self):
        """Worker thread for streaming"""
        while self.running and self.is_connected:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
            else:
                print("Failed to read frame")
                time.sleep(0.1)
    
    def get_frame(self):
        """Get the current frame"""
        return self.current_frame
    
    def save_frame(self, filename=None):
        """Save current frame to file"""
        if self.current_frame is not None:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"runcam_capture_{timestamp}.jpg"
            
            cv2.imwrite(filename, self.current_frame)
            print(f"Frame saved as {filename}")
            return True
        return False
    
    def stop(self):
        """Stop streaming and cleanup"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        if self.cap:
            self.cap.release()
        
        self.is_connected = False
        print("RunCam disconnected")

# Usage example
if __name__ == "__main__":
    runcam = RunCamWifiLink2()
    
    if runcam.connect():
        runcam.start_stream()
        
        try:
            while True:
                frame = runcam.get_frame()
                if frame is not None:
                    cv2.imshow('RunCam WifiLink 2', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    runcam.save_frame()
                    
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            runcam.stop()
            cv2.destroyAllWindows()