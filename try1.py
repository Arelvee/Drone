import cv2
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import queue
import time
import os
import sqlite3
from datetime import datetime
import csv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class YOLOPowerLineInspector:
    def __init__(self, root):
        self.root = root
        self.root.title("Real-Time UAV Inspection of Power Distribution Line Joints")
        self.root.geometry("1200x700")
        self.root.configure(bg='lightgray')
        
        # Application variables
        self.model = None
        self.cap = None
        self.running = False
        self.inspection_data = []
        
        # Define the specific class names/labels
        self.class_names = {
            0: "1-1A-Fired Wedge Joint-BARE",
            1: "1-1B-Fired Wedge Joint-COVERED", 
            2: "2-2A-Hummer Driven Wedge Joint-BARE",
            3: "2-2B-Hummer Driven Wedge Joint-COVERED"
        }
        
        # Thread management - USE QUEUES INSTEAD OF LOCKS FOR BETTER PERFORMANCE
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        
        # Performance optimization variables
        self.last_frame_time = 0
        self.target_fps = 30
        self.frame_interval = 1.0 / self.target_fps
        self.processing_enabled = True
        self.skip_frames = 0
        self.frame_skip_ratio = 2
        
        # Detection history
        self.current_detection = {
            "label": "No Detection", 
            "confidence": "0%", 
            "temperature": "0.0 °C",
            "timestamp": "",
            "multiple_detections": []  # New field for multiple detections
        }
        
        self.setup_gui()
        self.initialize_database()
        self.load_model()

    def initialize_database(self):
        """Initialize SQLite database with required tables"""
        try:
            self.conn = sqlite3.connect('power_line_inspection.db', check_same_thread=False)
            self.cursor = self.conn.cursor()
            
            # Create inspection records table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS inspection_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    defect_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    temperature TEXT NOT NULL,
                    distance TEXT,
                    line_number TEXT,
                    pole_number TEXT,
                    ambient_temperature TEXT,
                    weather_conditions TEXT,
                    inspector_name TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create class labels table and insert the predefined classes
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS class_labels (
                    id INTEGER PRIMARY KEY,
                    class_name TEXT NOT NULL UNIQUE,
                    description TEXT
                )
            ''')
            
            # Insert the predefined class names
            for class_id, class_name in self.class_names.items():
                self.cursor.execute('''
                    INSERT OR IGNORE INTO class_labels (id, class_name, description)
                    VALUES (?, ?, ?)
                ''', (class_id, class_name, f"Power line joint type: {class_name}"))
            
            self.conn.commit()
            logging.info("✅ Database initialized successfully")
            
        except sqlite3.Error as e:
            messagebox.showerror("Database Error", f"Failed to initialize database: {str(e)}")

    def setup_gui(self):
        # Title bar
        title_frame = tk.Frame(self.root, bg="navy", height=80)
        title_frame.pack(fill="x", padx=0, pady=0)
        title_frame.pack_propagate(False)
        
        title = tk.Label(title_frame, 
                        text="Real-Time UAV Inspection of Power Distribution Line Joints",
                        font=("Arial", 18, "bold"), 
                        bg="navy", 
                        fg="white")
        title.pack(expand=True)
        
        subtitle = tk.Label(title_frame,
                           text="AI-Powered Defect Detection System",
                           font=("Arial", 12),
                           bg="navy",
                           fg="lightblue")
        subtitle.pack(expand=True)

        # Main container
        main_container = tk.Frame(self.root, bg="lightgray")
        main_container.pack(fill="both", expand=True, padx=10, pady=10)

        # Left panel - Detection Results
        left_panel = tk.Frame(main_container, width=300, bg="white", relief="solid", bd=2)
        left_panel.pack(side="left", fill="y", padx=(0, 10))
        left_panel.pack_propagate(False)

        # Detection Results Section
        results_header = tk.Label(left_panel, text="Detection Results", 
                                 font=("Arial", 14, "bold"), 
                                 bg="navy", fg="white", pady=8)
        results_header.pack(fill="x", pady=(0, 10))

        # Current Detection Frame
        current_detection_frame = tk.LabelFrame(left_panel, text="Current Detection", 
                                               font=("Arial", 12, "bold"),
                                               bg="white", padx=10, pady=10)
        current_detection_frame.pack(fill="x", padx=10, pady=5)

        # Detection information with better styling - MODIFIED FOR MULTIPLE DETECTIONS
        self.detection_vars = {
            "label": tk.StringVar(value="No Detection"),
            "confidence": tk.StringVar(value="0%"),
            "temperature": tk.StringVar(value="0.0 °C"),
            "timestamp": tk.StringVar(value="--"),
            "multiple_info": tk.StringVar(value="")  # New variable for multiple detections info
        }

        info_styles = {"font": ("Arial", 11), "bg": "white", "anchor": "w", "pady": 3}

        tk.Label(current_detection_frame, text="Defect Type:", **info_styles).pack(fill="x")
        tk.Label(current_detection_frame, textvariable=self.detection_vars["label"], 
                font=("Arial", 11, "bold"), bg="white", fg="red").pack(fill="x")

        tk.Label(current_detection_frame, text="Confidence Level:", **info_styles).pack(fill="x")
        tk.Label(current_detection_frame, textvariable=self.detection_vars["confidence"], 
                font=("Arial", 11, "bold"), bg="white", fg="blue").pack(fill="x")

        tk.Label(current_detection_frame, text="Estimated Temperature:", **info_styles).pack(fill="x")
        tk.Label(current_detection_frame, textvariable=self.detection_vars["temperature"], 
                font=("Arial", 11, "bold"), bg="white", fg="darkorange").pack(fill="x")

        # NEW: Multiple Detections Information
        tk.Label(current_detection_frame, text="Multiple Detections:", **info_styles).pack(fill="x")
        multiple_info_label = tk.Label(current_detection_frame, textvariable=self.detection_vars["multiple_info"], 
                                     font=("Arial", 10), bg="white", fg="green", justify="left", wraplength=280)
        multiple_info_label.pack(fill="x")

        tk.Label(current_detection_frame, text="Time Detected:", **info_styles).pack(fill="x")
        tk.Label(current_detection_frame, textvariable=self.detection_vars["timestamp"], 
                **info_styles).pack(fill="x")

        # Performance Controls Section
        perf_frame = tk.LabelFrame(left_panel, text="Performance Controls", 
                                  font=("Arial", 12, "bold"),
                                  bg="white", padx=10, pady=10)
        perf_frame.pack(fill="x", padx=10, pady=5)

        # Frame skipping control
        skip_frame = tk.Frame(perf_frame, bg="white")
        skip_frame.pack(fill="x", pady=5)
        
        tk.Label(skip_frame, text="Frame Skip Ratio:", font=("Arial", 9), bg="white").pack(side="left")
        self.skip_var = tk.StringVar(value="2")
        skip_spinbox = ttk.Spinbox(skip_frame, from_=1, to=5, width=5, textvariable=self.skip_var,
                                  command=self.update_frame_skip)
        skip_spinbox.pack(side="right")
        
        # Processing toggle
        self.processing_var = tk.BooleanVar(value=True)
        processing_cb = ttk.Checkbutton(perf_frame, text="Enable Real-time Processing", 
                                       variable=self.processing_var,
                                       command=self.toggle_processing)
        processing_cb.pack(fill="x", pady=5)

        # Control Buttons Section
        control_frame = tk.LabelFrame(left_panel, text="Camera Controls", 
                                     font=("Arial", 12, "bold"),
                                     bg="white", padx=10, pady=10)
        control_frame.pack(fill="x", padx=10, pady=10)

        self.start_btn = tk.Button(control_frame, text="🚀 Start Inspection", 
                                  font=("Arial", 11, "bold"),
                                  bg="green3", fg="white",
                                  command=self.start_inspection,
                                  height=2)
        self.start_btn.pack(fill="x", pady=5)

        self.stop_btn = tk.Button(control_frame, text="🛑 Stop Inspection", 
                                 font=("Arial", 11, "bold"),
                                 bg="red3", fg="white",
                                 command=self.stop_inspection,
                                 state="disabled",
                                 height=2)
        self.stop_btn.pack(fill="x", pady=5)

        # Database Controls Section
        db_frame = tk.LabelFrame(left_panel, text="Database Controls", 
                                font=("Arial", 12, "bold"),
                                bg="white", padx=10, pady=10)
        db_frame.pack(fill="x", padx=10, pady=5)

        self.view_db_btn = tk.Button(db_frame, text="📊 View Records", 
                                    font=("Arial", 10),
                                    bg="purple", fg="white",
                                    command=self.view_database_records)
        self.view_db_btn.pack(fill="x", pady=3)

        self.clear_db_btn = tk.Button(db_frame, text="🗑️ Clear Records", 
                                     font=("Arial", 10),
                                     bg="darkred", fg="white",
                                     command=self.clear_database_records)
        self.clear_db_btn.pack(fill="x", pady=3)

        # Statistics Section
        stats_frame = tk.LabelFrame(left_panel, text="Inspection Statistics", 
                                   font=("Arial", 12, "bold"),
                                   bg="white", padx=10, pady=10)
        stats_frame.pack(fill="x", padx=10, pady=5)

        self.stats_vars = {
            "total_detections": tk.StringVar(value="0"),
            "session_time": tk.StringVar(value="00:00:00"),
            "frame_rate": tk.StringVar(value="0 FPS"),
            "processing_rate": tk.StringVar(value="0 FPS")
        }

        tk.Label(stats_frame, text="Total Detections:", **info_styles).pack(fill="x")
        tk.Label(stats_frame, textvariable=self.stats_vars["total_detections"], 
                font=("Arial", 11, "bold"), bg="white", fg="purple").pack(fill="x")

        tk.Label(stats_frame, text="Session Duration:", **info_styles).pack(fill="x")
        tk.Label(stats_frame, textvariable=self.stats_vars["session_time"], 
                **info_styles).pack(fill="x")

        tk.Label(stats_frame, text="Camera FPS:", **info_styles).pack(fill="x")
        tk.Label(stats_frame, textvariable=self.stats_vars["frame_rate"], 
                **info_styles).pack(fill="x")

        tk.Label(stats_frame, text="Processing FPS:", **info_styles).pack(fill="x")
        tk.Label(stats_frame, textvariable=self.stats_vars["processing_rate"], 
                **info_styles).pack(fill="x")

        # Center Panel - Camera Feed
        center_panel = tk.Frame(main_container, bg="black", relief="sunken", bd=3)
        center_panel.pack(side="left", fill="both", expand=True)

        self.camera_display = tk.Label(center_panel, 
                                      text="UAV Camera Feed\n\nClick 'Start Inspection' to begin\nreal-time power line inspection",
                                      font=("Arial", 14), 
                                      fg="white", bg="black",
                                      justify="center")
        self.camera_display.pack(expand=True, fill="both")

        # Right Panel - Inspection Details
        right_panel = tk.Frame(main_container, width=300, bg="white", relief="solid", bd=2)
        right_panel.pack(side="right", fill="y", padx=(10, 0))
        right_panel.pack_propagate(False)

        # Inspection Details Header
        details_header = tk.Label(right_panel, text="Inspection Details", 
                                 font=("Arial", 14, "bold"), 
                                 bg="navy", fg="white", pady=8)
        details_header.pack(fill="x", pady=(0, 10))

        # Input Form
        form_frame = tk.Frame(right_panel, bg="white", padx=15, pady=10)
        form_frame.pack(fill="both", expand=True)

        # Form fields with units and validation
        form_fields = [
            ("Distance from Target", "meters"),
            ("Line Number", ""),
            ("Pole Number", ""),
            ("Ambient Temperature", "°C"),
            ("Weather Conditions", ""),
            ("Inspector Name", "")
        ]

        self.form_entries = {}
        for i, (field, unit) in enumerate(form_fields):
            field_frame = tk.Frame(form_frame, bg="white")
            field_frame.pack(fill="x", pady=8)
            
            label_text = f"{field}:"
            if unit:
                label_text = f"{field} ({unit}):"
                
            tk.Label(field_frame, text=label_text, 
                    font=("Arial", 10, "bold"), 
                    bg="white", anchor="w").pack(fill="x")
            
            entry = ttk.Entry(field_frame, font=("Arial", 10))
            entry.pack(fill="x", pady=(5, 0))
            self.form_entries[field] = entry

        # Action Buttons
        button_frame = tk.Frame(right_panel, bg="white", pady=15)
        button_frame.pack(fill="x", side="bottom", padx=15)

        self.submit_btn = tk.Button(button_frame, text="📋 Save Inspection Report", 
                                   font=("Arial", 11, "bold"),
                                   bg="royalblue", fg="white",
                                   command=self.save_inspection_report)
        self.submit_btn.pack(fill="x", pady=5)

        self.export_btn = tk.Button(button_frame, text="💾 Export Data", 
                                   font=("Arial", 11),
                                   bg="steelblue", fg="white",
                                   command=self.export_data)
        self.export_btn.pack(fill="x", pady=5)

        # Status Bar
        self.status_var = tk.StringVar(value="🔴 System Ready - Load YOLO model to begin inspection")
        status_bar = tk.Label(self.root, textvariable=self.status_var, 
                             relief="sunken", anchor="w", 
                             font=("Arial", 10),
                             bg="lightcyan", fg="black")
        status_bar.pack(side="bottom", fill="x")

        # Statistics tracking
        self.detection_count = 0
        self.session_start_time = None
        self.frame_count = 0
        self.fps = 0
        self.processing_fps = 0
        self.last_fps_update = time.time()
        self.processing_frame_count = 0
        self.last_processing_time = time.time()

    def update_frame_skip(self):
        """Update frame skip ratio"""
        try:
            self.frame_skip_ratio = int(self.skip_var.get())
            self.status_var.set(f"🔄 Frame skip ratio set to {self.frame_skip_ratio}")
        except ValueError:
            self.frame_skip_ratio = 2
            self.skip_var.set("2")

    def toggle_processing(self):
        """Toggle real-time processing on/off"""
        self.processing_enabled = self.processing_var.get()
        status = "enabled" if self.processing_enabled else "disabled"
        self.status_var.set(f"🔄 Real-time processing {status}")

    # 🔹 Load YOLO Model
    def load_model(self):
        """Load YOLO model with error handling"""
        MODEL_PATH = 'finalsirbagsic.pt'
        
        if not os.path.exists(MODEL_PATH):
            logging.error(f"Model file '{MODEL_PATH}' not found!")
            self.status_var.set(f"❌ Model file '{MODEL_PATH}' not found")
            messagebox.showerror("Model Error", 
                               f"Model file '{MODEL_PATH}' not found!\n\n"
                               "Please ensure the YOLO model file is in the same directory.")
            return
        
        try:
            self.status_var.set("🔄 Loading YOLO model...")
            self.root.update()
            
            self.model = YOLO(MODEL_PATH)
            logging.info(f"✅ Loaded model from {MODEL_PATH}")
            
            # Update model class names with our specific classes
            if hasattr(self.model, 'names'):
                for i, name in self.class_names.items():
                    if i < len(self.model.names):
                        self.model.names[i] = name
            
            self.status_var.set("✅ YOLO Model Loaded Successfully - Ready for Inspection")
            
        except Exception as e:
            error_msg = f"❌ Error loading model: {e}"
            logging.error(error_msg)
            self.status_var.set(error_msg)
            messagebox.showerror("Model Loading Error", error_msg)

    # 🔹 Initialize Camera
    def open_camera(self):
        """Initialize camera with retry mechanism"""
        # Try different camera backends for better compatibility
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2, cv2.CAP_ANY]
        
        camera = None
        for backend in backends:
            camera = cv2.VideoCapture(0, backend)
            if camera.isOpened():
                break
            # Try different camera indices
            for i in range(1, 3):
                camera = cv2.VideoCapture(i, backend)
                if camera.isOpened():
                    break
            if camera and camera.isOpened():
                break
        
        if not camera or not camera.isOpened():
            retry_count = 0
            while retry_count < 5:
                logging.warning("🔄 Retrying camera connection...")
                time.sleep(2)
                camera = cv2.VideoCapture(0)
                if camera.isOpened():
                    break
                retry_count += 1

        if not camera or not camera.isOpened():
            logging.error("❌ Camera failed to open. Check connection.")
            return None

        # Camera configuration for better performance
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 30)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        logging.info("✅ Camera initialized successfully")
        return camera

    def start_inspection(self):
        """Start the camera and begin inspection"""
        try:
            if self.model is None:
                messagebox.showwarning("Model Not Loaded", 
                                     "Please wait for the YOLO model to load first.")
                return

            # Initialize camera using the patterned method
            self.cap = self.open_camera()
            if self.cap is None:
                messagebox.showerror("Camera Error", 
                                   "Cannot access camera. Please check:\n"
                                   "1. Camera is connected\n"
                                   "2. No other application is using the camera\n"
                                   "3. Camera drivers are installed")
                return

            self.running = True
            self.start_btn.config(state="disabled", bg="gray")
            self.stop_btn.config(state="normal", bg="red3")
            
            # Reset statistics
            self.detection_count = 0
            self.session_start_time = time.time()
            self.frame_count = 0
            self.processing_frame_count = 0
            self.last_fps_update = time.time()
            self.last_processing_time = time.time()
            
            # Clear queues
            self.clear_queues()
            
            self.status_var.set("🎥 Camera Active - Real-time inspection running...")

            # Start threads
            self.capture_thread = threading.Thread(target=self.capture_frames, daemon=True)
            self.processing_thread = threading.Thread(target=self.process_frames, daemon=True)
            
            self.capture_thread.start()
            self.processing_thread.start()
            
            # Start GUI updates
            self.update_session_timer()
            self.update_gui()

        except Exception as e:
            self.status_var.set(f"❌ Failed to start inspection: {str(e)}")
            messagebox.showerror("Startup Error", f"Cannot start inspection:\n{str(e)}")

    def stop_inspection(self):
        """Stop the inspection and release resources"""
        self.running = False
        self.start_btn.config(state="normal", bg="green3")
        self.stop_btn.config(state="disabled", bg="gray")
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear queues
        self.clear_queues()
        
        self.status_var.set("🛑 Inspection Stopped - Ready for new session")
        logging.info("🛑 Inspection stopped")

    def clear_queues(self):
        """Clear all queues"""
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

    # 🔹 Capture Frames from Camera Stream
    def capture_frames(self):
        """Capture frames from camera in separate thread"""
        while self.running and self.cap and self.cap.isOpened():
            try:
                success, frame = self.cap.read()
                if not success:
                    logging.error("❌ Failed to read from camera stream.")
                    time.sleep(0.1)
                    continue

                # Calculate FPS for camera feed
                current_time = time.time()
                self.frame_count += 1
                
                if current_time - self.last_fps_update >= 1.0:
                    self.fps = self.frame_count / (current_time - self.last_fps_update)
                    self.frame_count = 0
                    self.last_fps_update = current_time

                # Put frame in queue (non-blocking)
                if not self.frame_queue.full():
                    try:
                        self.frame_queue.put(frame.copy(), timeout=0.1)
                    except queue.Full:
                        pass  # Skip frame if queue is full
                
                # Control frame rate
                elapsed = time.time() - current_time
                sleep_time = self.frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                if self.running:
                    logging.error(f"Capture error: {e}")
                time.sleep(0.1)
                continue

    # 🔹 Process Frames with YOLO
    def process_frames(self):
        """Process frames with YOLO in separate thread"""
        processing_frame_count = 0
        
        while self.running:
            try:
                # Get frame from queue with timeout
                try:
                    frame = self.frame_queue.get(timeout=0.5)
                except queue.Empty:
                    continue

                # Frame skipping logic
                self.skip_frames = (self.skip_frames + 1) % self.frame_skip_ratio
                if self.skip_frames != 0 and self.frame_skip_ratio > 1:
                    # Skip processing this frame
                    if not self.result_queue.full():
                        self.result_queue.put((frame, self.current_detection))
                    continue
                
                if not self.processing_enabled:
                    # Skip YOLO processing
                    if not self.result_queue.full():
                        self.result_queue.put((frame, self.current_detection))
                    continue

                # 🔹 Run YOLO inference
                start_time = time.time()
                results = self.model(frame, conf=0.1, verbose=False, imgsz=320)
                processing_time = time.time() - start_time

                # Calculate processing FPS
                processing_frame_count += 1
                current_time = time.time()
                if current_time - self.last_processing_time >= 1.0:
                    self.processing_fps = processing_frame_count / (current_time - self.last_processing_time)
                    processing_frame_count = 0
                    self.last_processing_time = current_time

                # 🔹 Process detection results
                if results and len(results[0].boxes) > 0:
                    detection_info = self.extract_detection_info(results)
                    annotated_frame = results[0].plot()
                    
                    # Put result in queue
                    if not self.result_queue.full():
                        self.result_queue.put((annotated_frame, detection_info))
                else:
                    detection_info = {
                        "label": "No Detection",
                        "confidence": "0%", 
                        "temperature": "0.0 °C",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "multiple_info": ""
                    }
                    # Put original frame in queue
                    if not self.result_queue.full():
                        self.result_queue.put((frame, detection_info))
                
            except Exception as e:
                logging.error(f"❌ Error processing frame: {e}")
                continue

    def extract_detection_info(self, results):
        """Extract detection information from YOLO results - PROCESS ALL DETECTIONS"""
        detection_info = {
            "label": "No Detection",
            "confidence": "0%",
            "temperature": "0.0 °C",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "multiple_info": ""
        }
        
        try:
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes
                detections_count = len(boxes)
                
                # Process ALL detections, not just the highest confidence
                labels = []
                confidences = []
                class_ids = []
                temperatures = []
                
                for i in range(len(boxes)):
                    confidence = float(boxes.conf[i])
                    class_id = int(boxes.cls[i])
                    
                    # Use our predefined class names
                    class_name = self.class_names.get(class_id, f"Unknown Class {class_id}")
                    
                    # Calculate temperature for EACH detection
                    base_temp = 25.0
                    temp_increase = confidence * 15
                    temperature = base_temp + temp_increase
                    
                    labels.append(class_name)
                    confidences.append(confidence)
                    class_ids.append(class_id)
                    temperatures.append(temperature)
                
                # If multiple detections, combine them
                if len(labels) > 0:
                    if len(labels) == 1:
                        # Single detection
                        main_label = labels[0]
                        main_confidence = confidences[0]
                        main_temperature = temperatures[0]
                        multiple_info = ""
                    else:
                        # Multiple detections - show combined info
                        main_label = f"Multiple Detections ({len(labels)})"
                        main_confidence = max(confidences)  # Show highest confidence
                        main_temperature = max(temperatures)  # Show highest temperature
                        
                        # Create detailed multiple detection info
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
                        # Store all detections for database
                        "all_detections": list(zip(labels, confidences, class_ids, temperatures))
                    }
                    
                    # Store current detection
                    self.current_detection = detection_info
                    
                    # Count ALL detections, not just one
                    self.detection_count += len(labels)
                    
                    # Save ALL detections to database
                    self.save_detection_to_db(detection_info)
                    
        except Exception as e:
            logging.error(f"Error extracting detection info: {e}")
            
        return detection_info

    def update_gui(self):
        """Update GUI elements from main thread - NON-BLOCKING"""
        if self.running:
            try:
                # Get latest results from queue (non-blocking)
                if not self.result_queue.empty():
                    try:
                        annotated_frame, detection_info = self.result_queue.get_nowait()
                        
                        # Update detection information
                        for key, value in detection_info.items():
                            if key in self.detection_vars:
                                self.detection_vars[key].set(value)
                        
                        # Update camera display
                        self.display_annotated_frame(annotated_frame)
                        
                        # Store detection data
                        if detection_info["label"] != "No Detection":
                            self.inspection_data.append({
                                **detection_info,
                                "form_data": self.get_form_data()
                            })
                    
                    except queue.Empty:
                        pass
            
            except Exception as e:
                logging.error(f"GUI update error: {e}")
            
            # Update FPS displays
            self.stats_vars["frame_rate"].set(f"{self.fps:.1f} FPS")
            self.stats_vars["processing_rate"].set(f"{self.processing_fps:.1f} FPS")
            
            # Schedule next update
            self.root.after(30, self.update_gui)

    def display_annotated_frame(self, frame):
        """Display the annotated frame in the GUI"""
        try:
            # Convert frame for display
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Resize to fit display area while maintaining aspect ratio
            display_width = 640
            display_height = 480
            img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
            
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.camera_display.configure(image=imgtk, text="")
            self.camera_display.image = imgtk
            
        except Exception as e:
            logging.error(f"Frame display error: {e}")

    def update_session_timer(self):
        """Update session timer and statistics"""
        if self.running:
            # Update session time
            if self.session_start_time:
                elapsed = time.time() - self.session_start_time
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                seconds = int(elapsed % 60)
                self.stats_vars["session_time"].set(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            
            # Update detection count
            self.stats_vars["total_detections"].set(str(self.detection_count))
            
            # Schedule next update
            self.root.after(1000, self.update_session_timer)

    def get_form_data(self):
        """Get data from input form"""
        form_data = {}
        for field, entry in self.form_entries.items():
            form_data[field] = entry.get()
        return form_data

    def save_detection_to_db(self, detection_info):
        """Save ALL detections to SQLite database"""
        try:
            if detection_info["label"] != "No Detection" and "all_detections" in detection_info:
                form_data = self.get_form_data()
                
                # Save EACH detection individually
                for label, confidence, class_id, temperature in detection_info["all_detections"]:
                    self.cursor.execute('''
                        INSERT INTO inspection_records 
                        (timestamp, defect_type, confidence, temperature, distance, line_number, pole_number, ambient_temperature, weather_conditions, inspector_name)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        detection_info['timestamp'],
                        label,  # Use individual label
                        confidence,  # Use individual confidence
                        f"{temperature:.1f} °C",  # Use individual temperature
                        form_data.get('Distance from Target', ''),
                        form_data.get('Line Number', ''),
                        form_data.get('Pole Number', ''),
                        form_data.get('Ambient Temperature', ''),
                        form_data.get('Weather Conditions', ''),
                        form_data.get('Inspector Name', '')
                    ))
                    
                self.conn.commit()
                logging.info(f"✅ {len(detection_info['all_detections'])} detections saved to database")
                    
        except sqlite3.Error as e:
            logging.error(f"Database error: {e}")

    def save_inspection_data_manual(self):
        """Save inspection data manually without requiring detection"""
        try:
            form_data = self.get_form_data()
            
            # Validate required fields
            if not form_data.get('Line Number') or not form_data.get('Pole Number'):
                messagebox.showwarning("Missing Data", "Please enter Line Number and Pole Number")
                return
            
            # Get current detection info or create default
            if self.current_detection["label"] != "No Detection":
                detection_info = self.current_detection
            else:
                detection_info = {
                    "label": "Manual Inspection - No Defects",
                    "confidence": "0%",
                    "temperature": form_data.get('Ambient Temperature', '0.0 °C'),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "multiple_info": ""
                }
            
            # Save to database
            self.cursor.execute('''
                INSERT INTO inspection_records 
                (timestamp, defect_type, confidence, temperature, distance, line_number, pole_number, ambient_temperature, weather_conditions, inspector_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                detection_info['timestamp'],
                detection_info['label'],
                float(detection_info['confidence'].strip('%')) / 100.0 if detection_info['confidence'] != "0%" else 0.0,
                detection_info['temperature'],
                form_data.get('Distance from Target', ''),
                form_data.get('Line Number', ''),
                form_data.get('Pole Number', ''),
                form_data.get('Ambient Temperature', ''),
                form_data.get('Weather Conditions', ''),
                form_data.get('Inspector Name', '')
            ))
            
            self.conn.commit()
            
            messagebox.showinfo("Success", "Inspection data saved to database successfully!")
            self.status_var.set("✅ Inspection data saved to database")
            logging.info("✅ Manual inspection data saved to database")
            
        except sqlite3.Error as e:
            error_msg = f"Database error: {e}"
            messagebox.showerror("Database Error", error_msg)
            self.status_var.set("❌ Error saving to database")
        except Exception as e:
            error_msg = f"Error saving inspection data: {e}"
            messagebox.showerror("Save Error", error_msg)
            self.status_var.set("❌ Error saving inspection data")

    def save_inspection_report(self):
        """Save inspection report with current data"""
        try:
            # First save current data to database
            self.save_inspection_data_manual()
            
            # Then create the text report
            if not self.inspection_data:
                messagebox.showinfo("No Data", "No inspection data available to generate report.")
                return
            
            form_data = self.get_form_data()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"inspection_report_{timestamp}.txt"
            
            with open(filename, 'w') as f:
                f.write("UAV Power Line Inspection Report\n")
                f.write("=" * 50 + "\n")
                f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write("Inspection Details:\n")
                f.write("-" * 20 + "\n")
                for field, value in form_data.items():
                    f.write(f"{field}: {value}\n")
                
                f.write(f"\nTotal Defects Detected: {self.detection_count}\n")
                f.write(f"Session Duration: {self.stats_vars['session_time'].get()}\n\n")
                
                f.write("Defect Log:\n")
                f.write("-" * 20 + "\n")
                for i, detection in enumerate(self.inspection_data[-50:], 1):  # Last 50 detections
                    f.write(f"{i}. {detection['label']} - {detection['confidence']} ")
                    f.write(f"at {detection['timestamp']} (Temp: {detection['temperature']})\n")
            
            messagebox.showinfo("Report Saved", f"Inspection report saved as:\n{filename}")
            self.status_var.set(f"✅ Report saved: {filename}")
            logging.info(f"✅ Inspection report saved: {filename}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save report:\n{str(e)}")
            logging.error(f"❌ Failed to save report: {e}")

    def export_data(self):
        """Export data to CSV format from database"""
        try:
            # Get all records from database
            self.cursor.execute('''
                SELECT timestamp, defect_type, confidence, temperature, distance, 
                       line_number, pole_number, ambient_temperature, weather_conditions, inspector_name
                FROM inspection_records
                ORDER BY timestamp
            ''')
            records = self.cursor.fetchall()
            
            if not records:
                messagebox.showinfo("No Data", "No inspection data to export.")
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"inspection_data_{timestamp}.csv"
            
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "DefectType", "Confidence", "Temperature", "Distance", 
                               "LineNumber", "PoleNumber", "AmbientTemperature", "WeatherConditions", "InspectorName"])
                
                for record in records:
                    writer.writerow(record)
            
            messagebox.showinfo("Data Exported", f"Inspection data exported as:\n{filename}\n\nTotal records: {len(records)}")
            self.status_var.set(f"💾 Data exported: {filename} ({len(records)} records)")
            logging.info(f"💾 Data exported: {filename} ({len(records)} records)")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data:\n{str(e)}")
            logging.error(f"❌ Failed to export data: {e}")

    def view_database_records(self):
        """Display database records in a new window"""
        try:
            self.cursor.execute('''
                SELECT timestamp, defect_type, confidence, temperature, 
                       line_number, pole_number, inspector_name
                FROM inspection_records 
                ORDER BY timestamp DESC 
                LIMIT 100
            ''')
            records = self.cursor.fetchall()
            
            # Create new window
            db_window = tk.Toplevel(self.root)
            db_window.title("Inspection Records - Last 100 Detections")
            db_window.geometry("1000x500")
            db_window.configure(bg='white')
            
            # Title
            title_label = tk.Label(db_window, 
                                  text="Power Line Inspection Database Records",
                                  font=("Arial", 16, "bold"),
                                  bg="navy", fg="white",
                                  pady=10)
            title_label.pack(fill="x", padx=0, pady=(0, 10))
            
            # Create frame for treeview and scrollbar
            tree_frame = tk.Frame(db_window)
            tree_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Create treeview with more columns
            columns = ("Timestamp", "Defect Type", "Confidence", "Temperature", "Line Number", "Pole Number", "Inspector")
            tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=20)
            
            # Define headings
            tree.heading("Timestamp", text="Timestamp")
            tree.heading("Defect Type", text="Defect Type")
            tree.heading("Confidence", text="Confidence")
            tree.heading("Temperature", text="Temperature")
            tree.heading("Line Number", text="Line Number")
            tree.heading("Pole Number", text="Pole Number")
            tree.heading("Inspector", text="Inspector")
            
            # Define column widths
            tree.column("Timestamp", width=150, anchor="center")
            tree.column("Defect Type", width=200, anchor="w")
            tree.column("Confidence", width=100, anchor="center")
            tree.column("Temperature", width=120, anchor="center")
            tree.column("Line Number", width=100, anchor="center")
            tree.column("Pole Number", width=100, anchor="center")
            tree.column("Inspector", width=120, anchor="w")
            
            # Add scrollbars
            v_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
            h_scrollbar = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
            
            # Pack treeview and scrollbars
            tree.grid(row=0, column=0, sticky="nsew")
            v_scrollbar.grid(row=0, column=1, sticky="ns")
            h_scrollbar.grid(row=1, column=0, sticky="ew")
            
            # Configure grid weights
            tree_frame.grid_rowconfigure(0, weight=1)
            tree_frame.grid_columnconfigure(0, weight=1)
            
            # Insert records
            for record in records:
                # Convert confidence from decimal to percentage
                confidence = float(record[2]) * 100
                formatted_record = (
                    record[0],  # timestamp
                    record[1],  # defect_type
                    f"{confidence:.1f}%",  # confidence as percentage
                    record[3],  # temperature
                    record[4] if record[4] else "N/A",  # line_number
                    record[5] if record[5] else "N/A",  # pole_number
                    record[6] if record[6] else "N/A"   # inspector_name
                )
                tree.insert("", "end", values=formatted_record)
            
            # Add record count
            count_label = tk.Label(db_window, 
                                  text=f"Total records displayed: {len(records)}",
                                  font=("Arial", 10, "bold"),
                                  bg="white", fg="navy")
            count_label.pack(pady=5)
            
            # Add buttons frame
            button_frame = tk.Frame(db_window, bg="white")
            button_frame.pack(fill="x", padx=10, pady=10)
            
            refresh_btn = tk.Button(button_frame, text="🔄 Refresh", 
                                   font=("Arial", 10),
                                   bg="green", fg="white",
                                   command=lambda: self.refresh_records(tree, count_label))
            refresh_btn.pack(side="left", padx=5)
            
            export_btn = tk.Button(button_frame, text="💾 Export All to CSV", 
                                  font=("Arial", 10),
                                  bg="blue", fg="white",
                                  command=self.export_all_to_csv)
            export_btn.pack(side="left", padx=5)
            
            close_btn = tk.Button(button_frame, text="❌ Close", 
                                 font=("Arial", 10),
                                 bg="red", fg="white",
                                 command=db_window.destroy)
            close_btn.pack(side="right", padx=5)
            
            # Center the window
            db_window.update_idletasks()
            width = db_window.winfo_width()
            height = db_window.winfo_height()
            x = (db_window.winfo_screenwidth() // 2) - (width // 2)
            y = (db_window.winfo_screenheight() // 2) - (height // 2)
            db_window.geometry(f"+{x}+{y}")
            
        except sqlite3.Error as e:
            messagebox.showerror("Database Error", f"Failed to read records: {str(e)}")
            logging.error(f"❌ Database error reading records: {e}")

    def refresh_records(self, tree, count_label):
        """Refresh the records in the treeview"""
        try:
            # Clear existing items
            for item in tree.get_children():
                tree.delete(item)
            
            # Fetch updated records
            self.cursor.execute('''
                SELECT timestamp, defect_type, confidence, temperature, 
                       line_number, pole_number, inspector_name
                FROM inspection_records 
                ORDER BY timestamp DESC 
                LIMIT 100
            ''')
            records = self.cursor.fetchall()
            
            # Insert updated records
            for record in records:
                confidence = float(record[2]) * 100
                formatted_record = (
                    record[0],
                    record[1],
                    f"{confidence:.1f}%",
                    record[3],
                    record[4] if record[4] else "N/A",
                    record[5] if record[5] else "N/A",
                    record[6] if record[6] else "N/A"
                )
                tree.insert("", "end", values=formatted_record)
            
            # Update count label
            count_label.config(text=f"Total records displayed: {len(records)}")
            
        except sqlite3.Error as e:
            messagebox.showerror("Database Error", f"Failed to refresh records: {str(e)}")

    def export_all_to_csv(self):
        """Export all database records to CSV"""
        try:
            self.cursor.execute('''
                SELECT timestamp, defect_type, confidence, temperature, 
                       distance, line_number, pole_number, ambient_temperature, 
                       weather_conditions, inspector_name
                FROM inspection_records
                ORDER BY timestamp
            ''')
            records = self.cursor.fetchall()
            
            if not records:
                messagebox.showinfo("No Data", "No inspection data to export.")
                return
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"complete_inspection_data_{timestamp}.csv"
            
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow(["Timestamp", "DefectType", "Confidence", "Temperature", "Distance", 
                               "LineNumber", "PoleNumber", "AmbientTemperature", "WeatherConditions", "InspectorName"])
                
                # Write data
                for record in records:
                    # Convert confidence from decimal to percentage for display
                    confidence_pct = float(record[2]) * 100
                    formatted_record = [
                        record[0],  # timestamp
                        record[1],  # defect_type
                        f"{confidence_pct:.1f}%",  # confidence as percentage
                        record[3],  # temperature
                        record[4] if record[4] else "",  # distance
                        record[5] if record[5] else "",  # line_number
                        record[6] if record[6] else "",  # pole_number
                        record[7] if record[7] else "",  # ambient_temperature
                        record[8] if record[8] else "",  # weather_conditions
                        record[9] if record[9] else ""   # inspector_name
                    ]
                    writer.writerow(formatted_record)
            
            messagebox.showinfo("Export Successful", 
                              f"All inspection data exported successfully!\n\n"
                              f"File: {filename}\n"
                              f"Total records: {len(records)}")
            self.status_var.set(f"💾 All data exported: {filename} ({len(records)} records)")
            logging.info(f"💾 All data exported: {filename} ({len(records)} records)")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data:\n{str(e)}")
            logging.error(f"❌ Failed to export all data: {e}")

    def clear_database_records(self):
        """Clear all inspection records from database"""
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to delete ALL inspection records?\nThis action cannot be undone."):
            try:
                self.cursor.execute("DELETE FROM inspection_records")
                self.conn.commit()
                messagebox.showinfo("Success", "All inspection records have been cleared.")
                self.status_var.set("🗑️ All inspection records cleared")
                logging.info("🗑️ All inspection records cleared")
            except sqlite3.Error as e:
                messagebox.showerror("Database Error", f"Failed to clear records: {str(e)}")
                logging.error(f"❌ Failed to clear database records: {e}")

    def on_closing(self):
        """Handle application closing"""
        self.running = False
        if self.cap:
            self.cap.release()
        if hasattr(self, 'conn'):
            self.conn.close()
        self.root.destroy()
        logging.info("🔴 Application closed")

def main():
    root = tk.Tk()
    app = YOLOPowerLineInspector(root)
    
    # Center the window on screen
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"+{x}+{y}")

    # Handle window close event
    root.protocol("WM_DELETE_WINDOW", app.on_closing)

    root.mainloop()

if __name__ == "__main__":
    main()