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
import numpy as np

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
        
        # Thread management
        self.frame_queue = queue.Queue(maxsize=3)  # Increased queue size for better performance
        self.result_queue = queue.Queue(maxsize=3)
        
        # Performance optimization variables
        self.last_frame_time = 0
        self.target_fps = 30
        self.frame_interval = 1.0 / self.target_fps
        self.processing_enabled = True
        self.skip_frames = 0
        self.frame_skip_ratio = 2
        
        # Detection optimization variables
        self.confidence_threshold = 0.2  # Lower threshold to detect more defects
        self.iou_threshold = 0.4  # IOU threshold for NMS
        self.enable_enhancement = True  # Enable image enhancement
        
        # Detection history
        self.current_detections = []  # Changed to list to handle multiple detections
        self.detection_history = []
        
        self.setup_gui()
        self.initialize_database()
        self.load_model()

    def initialize_database(self):
        """Initialize SQLite database with required tables"""
        try:
            self.conn = sqlite3.connect('power_line_inspection.db')
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
                    bbox_coordinates TEXT,
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
            print("✅ Database initialized successfully")
            
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
                           text="AI-Powered Defect Detection System - Enhanced Detection Mode",
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

        # Current Detection Frame - Modified for multiple detections
        current_detection_frame = tk.LabelFrame(left_panel, text="Current Detections", 
                                               font=("Arial", 12, "bold"),
                                               bg="white", padx=10, pady=10)
        current_detection_frame.pack(fill="x", padx=10, pady=5)

        # Create a scrollable frame for multiple detections
        detection_canvas = tk.Canvas(current_detection_frame, bg="white", height=150)
        scrollbar = ttk.Scrollbar(current_detection_frame, orient="vertical", command=detection_canvas.yview)
        self.detection_scrollable_frame = ttk.Frame(detection_canvas)

        self.detection_scrollable_frame.bind(
            "<Configure>",
            lambda e: detection_canvas.configure(scrollregion=detection_canvas.bbox("all"))
        )

        detection_canvas.create_window((0, 0), window=self.detection_scrollable_frame, anchor="nw")
        detection_canvas.configure(yscrollcommand=scrollbar.set)

        detection_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Detection Controls Section
        detection_controls_frame = tk.LabelFrame(left_panel, text="Detection Controls", 
                                               font=("Arial", 12, "bold"),
                                               bg="white", padx=10, pady=10)
        detection_controls_frame.pack(fill="x", padx=10, pady=5)

        # Confidence threshold control
        confidence_frame = tk.Frame(detection_controls_frame, bg="white")
        confidence_frame.pack(fill="x", pady=5)
        
        tk.Label(confidence_frame, text="Confidence Threshold:", font=("Arial", 9), bg="white").pack(side="left")
        self.confidence_var = tk.StringVar(value="0.3")
        confidence_scale = ttk.Scale(confidence_frame, from_=0.1, to=0.9, 
                                    orient="horizontal", variable=self.confidence_var,
                                    command=self.update_confidence_threshold)
        confidence_scale.pack(side="right", fill="x", expand=True)
        
        self.confidence_label = tk.Label(confidence_frame, text="0.3", font=("Arial", 9), bg="white")
        self.confidence_label.pack(side="right", padx=(5, 0))

        # IOU threshold control
        iou_frame = tk.Frame(detection_controls_frame, bg="white")
        iou_frame.pack(fill="x", pady=5)
        
        tk.Label(iou_frame, text="IOU Threshold:", font=("Arial", 9), bg="white").pack(side="left")
        self.iou_var = tk.StringVar(value="0.4")
        iou_scale = ttk.Scale(iou_frame, from_=0.1, to=0.9, 
                             orient="horizontal", variable=self.iou_var,
                             command=self.update_iou_threshold)
        iou_scale.pack(side="right", fill="x", expand=True)
        
        self.iou_label = tk.Label(iou_frame, text="0.4", font=("Arial", 9), bg="white")
        self.iou_label.pack(side="right", padx=(5, 0))

        # Enhancement controls
        enhancement_frame = tk.Frame(detection_controls_frame, bg="white")
        enhancement_frame.pack(fill="x", pady=5)
        
        self.enhancement_var = tk.BooleanVar(value=True)
        enhancement_cb = ttk.Checkbutton(enhancement_frame, text="Enable Image Enhancement", 
                                        variable=self.enhancement_var,
                                        command=self.toggle_enhancement)
        enhancement_cb.pack(side="left")

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
            "processing_rate": tk.StringVar(value="0 FPS"),
            "current_detections": tk.StringVar(value="0")
        }

        info_styles = {"font": ("Arial", 11), "bg": "white", "anchor": "w", "pady": 3}

        tk.Label(stats_frame, text="Total Detections:", **info_styles).pack(fill="x")
        tk.Label(stats_frame, textvariable=self.stats_vars["total_detections"], 
                font=("Arial", 11, "bold"), bg="white", fg="purple").pack(fill="x")

        tk.Label(stats_frame, text="Current Frame Detections:", **info_styles).pack(fill="x")
        tk.Label(stats_frame, textvariable=self.stats_vars["current_detections"], 
                font=("Arial", 11, "bold"), bg="white", fg="green").pack(fill="x")

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

    def update_confidence_threshold(self, value):
        """Update confidence threshold for YOLO detection"""
        try:
            new_threshold = float(value)
            if 0.1 <= new_threshold <= 0.9:
                self.confidence_threshold = new_threshold
                self.confidence_label.config(text=f"{new_threshold:.1f}")
                self.status_var.set(f"🔄 Confidence threshold set to {new_threshold:.1f}")
        except ValueError:
            pass

    def update_iou_threshold(self, value):
        """Update IOU threshold for NMS"""
        try:
            new_threshold = float(value)
            if 0.1 <= new_threshold <= 0.9:
                self.iou_threshold = new_threshold
                self.iou_label.config(text=f"{new_threshold:.1f}")
                self.status_var.set(f"🔄 IOU threshold set to {new_threshold:.1f}")
        except ValueError:
            pass

    def toggle_enhancement(self):
        """Toggle image enhancement on/off"""
        self.enable_enhancement = self.enhancement_var.get()
        status = "enabled" if self.enable_enhancement else "disabled"
        self.status_var.set(f"🔄 Image enhancement {status}")

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

    def enhance_image(self, image):
        """Apply image enhancement techniques to improve detection"""
        if not self.enable_enhancement:
            return image
        
        try:
            # Convert to LAB color space for better contrast enhancement
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel for contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l_enhanced = clahe.apply(l)
            
            # Merge enhanced L channel with original A and B channels
            lab_enhanced = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            
            # Apply slight sharpening
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            enhanced = cv2.filter2D(enhanced, -1, kernel)
            
            return enhanced
        except Exception as e:
            print(f"Image enhancement error: {e}")
            return image

    def load_model(self):
        """Load YOLO model with error handling"""
        try:
            self.status_var.set("🔄 Loading YOLO model...")
            self.root.update()
            
            if not os.path.exists('sirbagsic.pt'):
                messagebox.showerror("Model Error", 
                                   "Model file 'sirbagsic.pt' not found!\n\n"
                                   "Please ensure the YOLO model file is in the same directory.")
                self.status_var.set("🔴 Error: Model file 'sirbagsic.pt' not found")
                return
            
            self.model = YOLO('sirbagsic.pt')
            
            # Update model class names with our specific classes
            if hasattr(self.model, 'names'):
                # Map the model's class names to our predefined names
                for i, name in self.class_names.items():
                    if i < len(self.model.names):
                        self.model.names[i] = name
            
            self.status_var.set("✅ YOLO Model Loaded Successfully - Ready for Inspection")
            
        except Exception as e:
            error_msg = f"❌ Failed to load YOLO model: {str(e)}"
            self.status_var.set(error_msg)
            messagebox.showerror("Model Loading Error", error_msg)

    def start_inspection(self):
        """Start the camera and begin inspection"""
        try:
            if self.model is None:
                messagebox.showwarning("Model Not Loaded", 
                                     "Please wait for the YOLO model to load first.")
                return

            # Try different camera backends for better performance
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2, cv2.CAP_ANY]
            
            for backend in backends:
                self.cap = cv2.VideoCapture(0, backend)
                if self.cap.isOpened():
                    break
                # Try different camera indices
                for i in range(1, 3):
                    self.cap = cv2.VideoCapture(i, backend)
                    if self.cap.isOpened():
                        break
                if self.cap.isOpened():
                    break
            
            if not self.cap.isOpened():
                messagebox.showerror("Camera Error", 
                                   "Cannot access camera. Please check:\n"
                                   "1. Camera is connected\n"
                                   "2. No other application is using the camera\n"
                                   "3. Camera drivers are installed")
                return

            # Camera configuration for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for lower latency
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))  # Better compression

            self.running = True
            self.start_btn.config(state="disabled", bg="gray")
            self.stop_btn.config(state="normal", bg="red3")
            
            # Reset statistics
            self.detection_count = 0
            self.session_start_time = time.time()
            self.frame_count = 0
            self.processing_frame_count = 0
            self.last_fps_update = time.time()
            
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
        
        self.status_var.set("🛑 Inspection Stopped - Ready for new session")
        
        # Clear queues
        self.clear_queues()

    def capture_frames(self):
        """Capture frames from camera in separate thread with optimized performance"""
        while self.running and self.cap:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    self.status_var.set("⚠️ Camera frame capture failed")
                    time.sleep(0.1)  # Prevent busy waiting
                    continue
                
                # Calculate FPS for camera feed
                current_time = time.time()
                self.frame_count += 1
                
                if current_time - self.last_fps_update >= 1.0:
                    self.fps = self.frame_count / (current_time - self.last_fps_update)
                    self.frame_count = 0
                    self.last_fps_update = current_time
                
                # Put frame in queue (replace old frame if queue full)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put(frame)
                
                # Control frame rate to prevent overwhelming the system
                elapsed = time.time() - current_time
                if elapsed < self.frame_interval:
                    time.sleep(self.frame_interval - elapsed)
                
            except Exception as e:
                if self.running:  # Only log if we're supposed to be running
                    print(f"Capture error: {e}")
                time.sleep(0.1)  # Prevent busy waiting on error
                continue

    def process_frames(self):
        """Process frames with YOLO in separate thread with optimized performance"""
        last_processing_time = time.time()
        processing_frame_count = 0
        
        while self.running:
            try:
                # Get frame with timeout
                frame = self.frame_queue.get(timeout=0.5)
                
                # Frame skipping logic - only process every Nth frame
                self.skip_frames = (self.skip_frames + 1) % self.frame_skip_ratio
                if self.skip_frames != 0 and self.frame_skip_ratio > 1:
                    # Skip processing this frame, just display it
                    if self.result_queue.full():
                        try:
                            self.result_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.result_queue.put((frame, self.current_detections))
                    continue
                
                if not self.processing_enabled:
                    # Skip YOLO processing, just pass the frame through
                    if self.result_queue.full():
                        try:
                            self.result_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.result_queue.put((frame, self.current_detections))
                    continue
                
                # Apply image enhancement
                enhanced_frame = self.enhance_image(frame)
                
                # Run YOLO inference with optimized settings for maximum detection
                start_time = time.time()
                results = self.model(enhanced_frame, 
                                   conf=self.confidence_threshold,  # Dynamic confidence
                                   iou=self.iou_threshold,         # Dynamic IOU
                                   verbose=False, 
                                   imgsz=320,
                                   augment=True)  # Enable augmentation for better detection
                processing_time = time.time() - start_time
                
                # Calculate processing FPS
                processing_frame_count += 1
                current_time = time.time()
                if current_time - last_processing_time >= 1.0:
                    self.processing_fps = processing_frame_count / (current_time - last_processing_time)
                    processing_frame_count = 0
                    last_processing_time = current_time
                
                annotated_frame = results[0].plot()
                
                # Extract detection information
                detection_info = self.extract_detection_info(results)
                
                # Put result in queue
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.result_queue.put((annotated_frame, detection_info))
                
            except queue.Empty:
                continue
            except Exception as e:
                if self.running:
                    print(f"Processing error: {e}")
                continue

    def extract_detection_info(self, results):
        """Extract detection information from YOLO results - now handles multiple detections"""
        all_detections = []
        
        try:
            if len(results[0].boxes) > 0:
                boxes = results[0].boxes
                
                for i, box in enumerate(boxes):
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    bbox = box.xyxy[0].tolist()  # Get bounding box coordinates
                    
                    # Use our predefined class names
                    class_name = self.class_names.get(class_id, f"Unknown Class {class_id}")
                    
                    # Simulate temperature based on confidence and defect type
                    base_temp = 25.0  # Ambient temperature
                    temp_increase = confidence * 15  # More confidence = higher estimated temperature
                    
                    detection_info = {
                        "label": class_name,
                        "confidence": f"{confidence:.1%}",
                        "temperature": f"{base_temp + temp_increase:.1f} °C",
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "bbox": bbox
                    }
                    
                    all_detections.append(detection_info)
                    
                    # Save to database immediately with form data
                    self.save_detection_to_db(detection_info)
                
                # Store current detections
                self.current_detections = all_detections
                
                # Count detections
                self.detection_count += len(all_detections)
                
        except Exception as e:
            print(f"Error extracting detection info: {e}")
            
        return all_detections

    def save_detection_to_db(self, detection_info):
        """Save detection to SQLite database"""
        try:
            if detection_info["label"] != "No Detection":
                form_data = self.get_form_data()
                
                self.cursor.execute('''
                    INSERT INTO inspection_records 
                    (timestamp, defect_type, confidence, temperature, distance, line_number, pole_number, ambient_temperature, weather_conditions, inspector_name, bbox_coordinates)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    detection_info['timestamp'],
                    detection_info['label'],
                    float(detection_info['confidence'].strip('%')) / 100.0,
                    detection_info['temperature'],
                    form_data.get('Distance from Target', ''),
                    form_data.get('Line Number', ''),
                    form_data.get('Pole Number', ''),
                    form_data.get('Ambient Temperature', ''),
                    form_data.get('Weather Conditions', ''),
                    form_data.get('Inspector Name', ''),
                    str(detection_info.get('bbox', []))
                ))
                
                self.conn.commit()
                
        except sqlite3.Error as e:
            print(f"Database error: {e}")

    def update_detection_display(self, detections):
        """Update the detection display with multiple detections"""
        # Clear previous detections
        for widget in self.detection_scrollable_frame.winfo_children():
            widget.destroy()
        
        if not detections:
            # Show no detection message
            no_detect_label = tk.Label(self.detection_scrollable_frame, 
                                     text="No Defects Detected",
                                     font=("Arial", 10, "italic"),
                                     bg="white", fg="gray")
            no_detect_label.pack(fill="x", pady=2)
            return
        
        # Display each detection
        for i, detection in enumerate(detections):
            detection_frame = tk.Frame(self.detection_scrollable_frame, bg="white", relief="solid", bd=1)
            detection_frame.pack(fill="x", pady=2, padx=2)
            
            # Detection header
            header_frame = tk.Frame(detection_frame, bg="lightyellow")
            header_frame.pack(fill="x")
            
            tk.Label(header_frame, text=f"Defect #{i+1}: {detection['label']}", 
                    font=("Arial", 9, "bold"), bg="lightyellow", fg="black").pack(side="left")
            
            # Detection details
            details_frame = tk.Frame(detection_frame, bg="white")
            details_frame.pack(fill="x", padx=5)
            
            tk.Label(details_frame, text=f"Confidence: {detection['confidence']}", 
                    font=("Arial", 8), bg="white", fg="blue").pack(anchor="w")
            tk.Label(details_frame, text=f"Temperature: {detection['temperature']}", 
                    font=("Arial", 8), bg="white", fg="darkorange").pack(anchor="w")
            tk.Label(details_frame, text=f"Time: {detection['timestamp']}", 
                    font=("Arial", 7), bg="white", fg="gray").pack(anchor="w")

    def update_gui(self):
        """Update GUI elements from main thread with performance optimization"""
        if self.running:
            try:
                # Get latest results from queue (non-blocking)
                if not self.result_queue.empty():
                    annotated_frame, detections = self.result_queue.get_nowait()
                    
                    # Update detection information display
                    self.update_detection_display(detections)
                    
                    # Update current detections count
                    self.stats_vars["current_detections"].set(str(len(detections)))
                    
                    # Update camera display
                    self.display_annotated_frame(annotated_frame)
                    
                    # Store detection data
                    if detections:
                        for detection in detections:
                            self.inspection_data.append({
                                **detection,
                                "form_data": self.get_form_data()
                            })
            
            except queue.Empty:
                pass
            except Exception as e:
                print(f"GUI update error: {e}")
            
            # Update FPS displays
            self.stats_vars["frame_rate"].set(f"{self.fps:.1f} FPS")
            self.stats_vars["processing_rate"].set(f"{self.processing_fps:.1f} FPS")
            self.stats_vars["total_detections"].set(str(self.detection_count))
            
            # Schedule next update
            self.root.after(30, self.update_gui)

    def display_annotated_frame(self, frame):
        """Display the annotated frame in the GUI with optimized performance"""
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
            print(f"Frame display error: {e}")

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
            
            # Schedule next update
            self.root.after(1000, self.update_session_timer)

    def get_form_data(self):
        """Get data from input form"""
        form_data = {}
        for field, entry in self.form_entries.items():
            form_data[field] = entry.get()
        return form_data

    def save_inspection_report(self):
        """Save inspection report with current data"""
        try:
            # First save current data to database
            self.save_inspection_data_manual()
            
            # Then create the text report (optional)
            if not self.inspection_data:
                return  # Already saved to DB, no need for text report
            
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
                for i, detection in enumerate(self.inspection_data, 1):
                    f.write(f"{i}. {detection['label']} - {detection['confidence']} ")
                    f.write(f"at {detection['timestamp']} (Temp: {detection['temperature']})\n")
            
            messagebox.showinfo("Report Saved", f"Inspection report saved as:\n{filename}")
            self.status_var.set(f"✅ Report saved: {filename}")
            
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save report:\n{str(e)}")

    def save_inspection_data_manual(self):
        """Save inspection data manually without requiring detection"""
        try:
            form_data = self.get_form_data()
            
            # Validate required fields
            if not form_data.get('Line Number') or not form_data.get('Pole Number'):
                messagebox.showwarning("Missing Data", "Please enter Line Number and Pole Number")
                return
            
            # Get current detection info or create default
            if self.current_detections:
                detection_info = self.current_detections[0]  # Use first detection
            else:
                detection_info = {
                    "label": "Manual Inspection - No Defects",
                    "confidence": "0%",
                    "temperature": form_data.get('Ambient Temperature', '0.0 °C'),
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "bbox": []
                }
            
            # Save to database
            self.cursor.execute('''
                INSERT INTO inspection_records 
                (timestamp, defect_type, confidence, temperature, distance, line_number, pole_number, ambient_temperature, weather_conditions, inspector_name, bbox_coordinates)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                form_data.get('Inspector Name', ''),
                str(detection_info.get('bbox', []))
            ))
            
            self.conn.commit()
            
            messagebox.showinfo("Success", "Inspection data saved to database successfully!")
            self.status_var.set("✅ Inspection data saved to database")
            
        except sqlite3.Error as e:
            error_msg = f"Database error: {e}"
            messagebox.showerror("Database Error", error_msg)
            self.status_var.set("❌ Error saving to database")
        except Exception as e:
            error_msg = f"Error saving inspection data: {e}"
            messagebox.showerror("Save Error", error_msg)
            self.status_var.set("❌ Error saving inspection data")

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
            
            with open(filename, 'w') as f:
                f.write("Timestamp,DefectType,Confidence,Temperature,Distance,LineNumber,PoleNumber,AmbientTemperature,WeatherConditions,InspectorName\n")
                for record in records:
                    f.write(','.join(f'"{str(item)}"' for item in record) + '\n')
            
            messagebox.showinfo("Data Exported", f"Inspection data exported as:\n{filename}\n\nTotal records: {len(records)}")
            self.status_var.set(f"💾 Data exported: {filename} ({len(records)} records)")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data:\n{str(e)}")

    def view_database_records(self):
        """Display database records in a new window"""
        try:
            self.cursor.execute('''
                SELECT timestamp, defect_type, confidence, temperature 
                FROM inspection_records 
                ORDER BY timestamp DESC 
                LIMIT 100
            ''')
            records = self.cursor.fetchall()
            
            # Create new window
            db_window = tk.Toplevel(self.root)
            db_window.title("Inspection Records - Last 100 Detections")
            db_window.geometry("800x400")
            
            # Create treeview
            tree = ttk.Treeview(db_window, columns=("Timestamp", "Defect Type", "Confidence", "Temperature"), show="headings")
            tree.heading("Timestamp", text="Timestamp")
            tree.heading("Defect Type", text="Defect Type")
            tree.heading("Confidence", text="Confidence")
            tree.heading("Temperature", text="Temperature")
            
            tree.column("Timestamp", width=150)
            tree.column("Defect Type", width=200)
            tree.column("Confidence", width=100)
            tree.column("Temperature", width=100)
            
            for record in records:
                tree.insert("", "end", values=record)
            
            tree.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Add scrollbar
            scrollbar = ttk.Scrollbar(db_window, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)
            scrollbar.pack(side="right", fill="y")
            
        except sqlite3.Error as e:
            messagebox.showerror("Database Error", f"Failed to read records: {str(e)}")

    def clear_database_records(self):
        """Clear all inspection records from database"""
        if messagebox.askyesno("Confirm Clear", "Are you sure you want to delete ALL inspection records?\nThis action cannot be undone."):
            try:
                self.cursor.execute("DELETE FROM inspection_records")
                self.conn.commit()
                messagebox.showinfo("Success", "All inspection records have been cleared.")
                self.status_var.set("🗑️ All inspection records cleared")
            except sqlite3.Error as e:
                messagebox.showerror("Database Error", f"Failed to clear records: {str(e)}")

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

    def on_closing(self):
        """Handle application closing"""
        self.running = False
        if self.cap:
            self.cap.release()
        if hasattr(self, 'conn'):
            self.conn.close()
        self.root.destroy()

def main():
    root = tk.Tk()
    app = YOLOPowerLineInspector(root)
    
    # Set window icon (optional)
    # root.iconbitmap('app_icon.ico')  # Uncomment if you have an icon file
    
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