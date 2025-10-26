import cv2
import logging
import time
import random
import queue
import threading
import os
import numpy as np
from datetime import datetime
from ultralytics import YOLO
from config import Config

class PowerLineDetector:
    """Handles YOLO model loading and frame processing - USES NATIVE YOLO BOUNDING BOXES"""
    
    def __init__(self):
        self.config = Config()
        self.model = None
        self.running = False
        self.processing_enabled = True
        
        # Frame management
        self.skip_factor = self.config.DETECTION_INTERVAL
        self.frame_counter = 0
        
        # Detection state - uses YOLO native format
        self.current_detection = {
            "label": "No Detection",
            "confidence": "0%", 
            "temperature": "0.0 °C",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "multiple_info": "",
            "all_detections": [],  # Will store raw YOLO detections
            "boxes_xyxy": []       # Will store YOLO bounding boxes in xyxy format
        }
        
        # Performance tracking
        self.processing_fps = 0
        self.processing_frame_count = 0
        self.last_processing_time = time.time()
        self.target_processing_time = 1.0 / self.config.TARGET_FPS
        
        # Load YOLO model
        self.load_model()
    
    def load_model(self):
        """Load YOLO model with proper configuration"""
        if not os.path.exists(self.config.MODEL_PATH):
            logging.error(f"Model file '{self.config.MODEL_PATH}' not found!")
            raise FileNotFoundError(f"Model file '{self.config.MODEL_PATH}' not found")
        
        try:
            logging.info("🔄 Loading YOLO model...")
            
            # Load the custom trained model
            self.model = YOLO(self.config.MODEL_PATH)
            
            # Use GPU if available and configured
            try:
                if self.config.USE_GPU:
                    self.model.to('cuda')
                    logging.info("✅ Model loaded on CUDA")
                else:
                    logging.info("✅ Model loaded on CPU")
            except Exception as gpu_error:
                logging.warning(f"GPU not available, using CPU: {gpu_error}")
                logging.info("✅ Model loaded on CPU")
            
            # Verify model classes
            if hasattr(self.model, 'names'):
                logging.info(f"📊 Model classes: {self.model.names}")
                # Ensure our class names match the model
                for i, name in self.config.CLASS_NAMES.items():
                    if i in self.model.names:
                        logging.info(f"✅ Class {i}: {self.model.names[i]} -> {name}")
                    else:
                        logging.warning(f"⚠️ Class {i} not found in model")
            else:
                logging.warning("⚠️ Model class names not available")
            
            logging.info(f"✅ YOLO model '{self.config.MODEL_PATH}' loaded successfully")
            
        except Exception as e:
            logging.error(f"❌ Error loading YOLO model: {e}")
            raise
    
    def process_frame(self, frame):
        """Process frame using YOLO model - RETURNS NATIVE YOLO BOUNDING BOXES"""
        start_time = time.time()
        
        if not self.processing_enabled:
            # Return original frame with current detection
            return frame, self.current_detection
        
        # Frame skipping for performance
        self.frame_counter += 1
        should_process = (self.frame_counter % self.skip_factor == 0)
        
        if should_process:
            try:
                # Perform YOLO inference - THIS IS WHERE BOUNDING BOXES ARE GENERATED
                detection_start = time.time()
                
                # Run YOLO inference - this returns the native bounding boxes
                results = self.model(
                    frame, 
                    conf=self.config.CONFIDENCE_THRESHOLD, 
                    imgsz=self.config.IMAGE_SIZE,
                    verbose=False,
                    max_det=self.config.MAX_CONCURRENT_DETECTIONS
                )
                
                detection_time = time.time() - detection_start
                logging.debug(f"YOLO detection time: {detection_time:.3f}s")
                
                # Process YOLO results to extract native bounding boxes
                if results and len(results) > 0:
                    detection_info = self.extract_yolo_detections(results[0])
                    self.current_detection = detection_info
                    
                    # Draw bounding boxes using YOLO's native coordinates
                    annotated_frame = self.draw_yolo_boxes(frame, detection_info)
                else:
                    # No detections
                    self.current_detection = self.get_no_detection_info()
                    annotated_frame = frame
                    
            except Exception as e:
                logging.error(f"❌ YOLO detection error: {e}")
                annotated_frame = frame
                detection_info = self.current_detection
        else:
            # Skip detection this frame
            annotated_frame = frame
            detection_info = self.current_detection
        
        # Control processing rate for smooth playback
        total_processing_time = time.time() - start_time
        sleep_time = self.target_processing_time - total_processing_time
        
        if sleep_time > 0:
            time.sleep(sleep_time)
        
        # Update processing FPS
        self.update_processing_fps(start_time)
        
        return annotated_frame, detection_info
    
    def extract_yolo_detections(self, result):
        """Extract detections from YOLO result - USES NATIVE YOLO BOUNDING BOXES"""
        detection_info = self.get_no_detection_info()
        
        try:
            # Check if we have any detections
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                
                labels, confidences, class_ids, temperatures, boxes_xyxy = [], [], [], [], []
                
                # Iterate through all detections from YOLO
                for i in range(len(boxes)):
                    try:
                        # Get confidence and class from YOLO
                        confidence = float(boxes.conf[i].item())
                        class_id = int(boxes.cls[i].item())
                        
                        # Get the class name from our configuration
                        class_name = self.config.CLASS_NAMES.get(class_id, f"Class_{class_id}")
                        
                        # Simulate temperature based on class and confidence
                        temperature = self.simulate_temperature(class_id, confidence)
                        
                        # GET NATIVE YOLO BOUNDING BOX COORDINATES (xyxy format)
                        # This is the actual bounding box from your trained model
                        box_coords = boxes.xyxy[i]  # [x1, y1, x2, y2]
                        x1, y1, x2, y2 = box_coords
                        
                        # Convert to integers for drawing
                        coords = (int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item()))
                        
                        # Store detection information
                        labels.append(class_name)
                        confidences.append(confidence)
                        class_ids.append(class_id)
                        temperatures.append(temperature)
                        boxes_xyxy.append(coords)
                        
                        logging.debug(f"🎯 YOLO Detection: {class_name} {confidence:.1%} at {coords}")
                        
                    except Exception as box_error:
                        logging.warning(f"⚠️ Error processing box {i}: {box_error}")
                        continue
                
                if labels:
                    if len(labels) == 1:
                        # Single detection
                        main_label = labels[0]
                        main_confidence = confidences[0]
                        main_temperature = temperatures[0]
                        multiple_info = ""
                    else:
                        # Multiple detections
                        main_label = f"Multiple Detections ({len(labels)})"
                        main_confidence = max(confidences)
                        main_temperature = max(temperatures)
                        
                        # Create detailed info
                        multiple_lines = []
                        for i, (label, conf, temp) in enumerate(zip(labels, confidences, temperatures), 1):
                            multiple_lines.append(f"{i}. {label}: {conf:.1%} - {temp:.1f}°C")
                        multiple_info = "\n".join(multiple_lines)
                    
                    detection_info = {
                        "label": main_label,
                        "confidence": f"{main_confidence:.1%}",
                        "temperature": f"{main_temperature:.1f} °C",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "multiple_info": multiple_info,
                        "all_detections": list(zip(labels, confidences, class_ids, temperatures)),
                        "boxes_xyxy": boxes_xyxy  # Native YOLO bounding boxes
                    }
                    
                    logging.info(f"✅ YOLO found {len(labels)} detection(s): {main_label}")
                    
        except Exception as e:
            logging.error(f"❌ Error extracting YOLO detections: {e}")
            
        return detection_info
    
    def draw_yolo_boxes(self, frame, detection_info):
        """Draw bounding boxes using YOLO's native coordinates"""
        annotated_frame = frame.copy()
        
        if "boxes_xyxy" in detection_info and "all_detections" in detection_info:
            boxes_xyxy = detection_info["boxes_xyxy"]
            all_dets = detection_info["all_detections"]
            
            for coords, det in zip(boxes_xyxy, all_dets):
                try:
                    x1, y1, x2, y2 = coords
                    label, conf, class_id, temp = det
                    
                    # DRAW BOUNDING BOX FROM YOLO COORDINATES
                    color = self.get_class_color(class_id)
                    
                    # Draw rectangle using YOLO bounding box coordinates
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw label with confidence
                    if self.config.SHOW_CONFIDENCE_SCORES:
                        label_text = f"{label} {conf:.0%}"
                    else:
                        label_text = f"{label}"
                    
                    # Calculate text background
                    text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
                    cv2.rectangle(annotated_frame, (x1, y1 - text_size[1] - 10), 
                                 (x1 + text_size[0], y1), color, -1)
                    
                    # Draw text
                    cv2.putText(annotated_frame, label_text, (x1, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                    
                    # Draw temperature
                    if self.config.ENABLE_TEMPERATURE_SIMULATION:
                        temp_text = f"{temp:.1f}°C"
                        temp_size = cv2.getTextSize(temp_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                        cv2.rectangle(annotated_frame, (x1, y2), 
                                     (x1 + temp_size[0], y2 + temp_size[1] + 5), (0, 200, 200), -1)
                        cv2.putText(annotated_frame, temp_text, (x1, y2 + temp_size[1] + 2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
                    
                except Exception as draw_error:
                    logging.warning(f"⚠️ Error drawing box: {draw_error}")
        
        return annotated_frame
    
    def get_class_color(self, class_id):
        """Get distinct color for each class"""
        colors = [
            (0, 255, 0),    # Green - Class 0
            (255, 0, 0),    # Blue - Class 1  
            (0, 0, 255),    # Red - Class 2
            (255, 255, 0),  # Cyan - Class 3
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        return colors[class_id % len(colors)]
    
    def simulate_temperature(self, class_id, confidence):
        """Generate realistic temperature based on class and confidence"""
        if not self.config.ENABLE_TEMPERATURE_SIMULATION:
            return 0.0
            
        temp_range = self.config.TEMPERATURE_RANGES.get(class_id, (30.0, 60.0))
        low, high = temp_range
        
        # Higher confidence = potentially higher temperature
        bias = confidence
        simulated_temp = random.uniform(
            low + (high - low) * bias * 0.1, 
            high - (high - low) * (1 - bias) * 0.05
        )
        
        return round(simulated_temp + random.uniform(-0.5, 0.5), 1)
    
    def update_processing_fps(self, start_time):
        """Update processing FPS calculation"""
        self.processing_frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_processing_time >= 0.5:
            self.processing_fps = self.processing_frame_count / (current_time - self.last_processing_time)
            self.processing_frame_count = 0
            self.last_processing_time = current_time
    
    def get_no_detection_info(self):
        """Return no detection information"""
        return {
            "label": "No Detection",
            "confidence": "0%", 
            "temperature": "0.0 °C",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "multiple_info": "",
            "all_detections": [],
            "boxes_xyxy": []
        }
    
    def set_processing_enabled(self, enabled):
        """Enable or disable processing"""
        self.processing_enabled = enabled
        if enabled:
            logging.info(f"🔄 YOLO processing ENABLED (Interval: {self.skip_factor})")
        else:
            logging.info("⚪ YOLO processing DISABLED")
            self.current_detection = self.get_no_detection_info()
    
    def set_detection_interval(self, interval):
        """Set detection interval"""
        self.skip_factor = max(1, min(10, interval))
        self.frame_counter = 0
        logging.info(f"🔧 YOLO detection interval: {self.skip_factor}")
    
    def get_processing_fps(self):
        """Get current processing FPS"""
        return self.processing_fps
    
    def get_detection_interval(self):
        """Get current detection interval"""
        return self.skip_factor
    
    def get_current_detection(self):
        """Get current detection data"""
        return self.current_detection
    
    def stop(self):
        """Stop the detector"""
        self.running = False
        logging.info("🛑 YOLO Detector stopped")