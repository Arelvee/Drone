import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import logging
import time
import threading
import cv2
from datetime import datetime
from config import Config
from database import DatabaseManager
from detector import PowerLineDetector
from camera import CameraManager

class PowerLineInspectorGUI:
    """Main GUI application for Power Line Inspection - RESPONSIVE VERSION"""
    
    def __init__(self, root):
        self.root = root
        self.config = Config()
        self.setup_gui()
        
        # Initialize components from your existing files
        self.database = DatabaseManager()
        self.detector = PowerLineDetector()
        self.camera = CameraManager()
        
        # Application state
        self.running = False
        self.inspection_data = []
        self.detection_count = 0
        self.session_start_time = None
        self.last_gui_update = time.time()
        self.current_frame = None
        
        # Make window responsive
        self.setup_responsive_layout()
        
        # Start GUI updates
        self.update_session_timer()
        self.update_time()
        
        # Set initial status
        self.status_var.set("✅ System Ready - Click 'Start Inspection' to begin")
        logging.info("🚀 Power Line Inspection System Initialized")

    def setup_responsive_layout(self):
        """Setup responsive grid layout"""
        # Configure root grid
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Bind resize event
        self.root.bind('<Configure>', self.on_window_resize)

    def on_window_resize(self, event):
        """Handle window resize event"""
        if event.widget == self.root:
            self.update_responsive_layout()

    def update_responsive_layout(self):
        """Update layout based on current window size"""
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        
        # Adjust fonts based on window size
        if width < 1000:
            self.font_scale = 0.8
        elif width < 1200:
            self.font_scale = 0.9
        else:
            self.font_scale = 1.0

    def setup_gui(self):
        """Setup the main GUI layout with responsive design"""
        self.root.title("Real-Time UAV Power Line Inspection System")
        self.root.geometry(f"{self.config.WINDOW_WIDTH}x{self.config.WINDOW_HEIGHT}")
        self.root.configure(bg='#f0f0f0')
        
        # Make window resizable with minimum size
        self.root.minsize(1000, 700)
        
        # Configure style
        self.setup_styles()
        
        # Title bar
        self.create_title_bar()
        
        # Main container with responsive grid
        main_container = tk.Frame(self.root, bg="#f0f0f0")
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Configure main container grid for responsiveness
        main_container.grid_rowconfigure(0, weight=1)
        main_container.grid_columnconfigure(1, weight=1)  # Center panel gets most space
        
        # Left panel - Detection Results and Controls
        self.create_left_panel(main_container)
        
        # Center panel - Camera Feed
        self.create_center_panel(main_container)
        
        # Right panel - Inspection Details
        self.create_right_panel(main_container)
        
        # Status bar
        self.create_status_bar()

    def setup_styles(self):
        """Setup modern styling for widgets with responsive fonts"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        self.colors = {
            'primary': '#2c3e50',
            'secondary': '#34495e',
            'accent': '#3498db',
            'success': '#27ae60',
            'warning': '#f39c12',
            'danger': '#e74c3c',
            'light': '#ecf0f1',
            'dark': '#2c3e50',
            'background': '#f8f9fa'
        }
        
        # Initial font scale
        self.font_scale = 1.0

    def get_font(self, size, weight="normal"):
        """Get responsive font based on current scale"""
        sizes = {
            'title': 20,
            'subtitle': 11,
            'header': 12,
            'button_large': 12,
            'button_medium': 11,
            'button_small': 10,
            'label': 10,
            'small': 9,
            'tiny': 8
        }
        base_size = sizes.get(size, 10)
        actual_size = int(base_size * self.font_scale)
        
        if weight == "bold":
            return ("Arial", actual_size, "bold")
        return ("Arial", actual_size)

    def create_title_bar(self):
        """Create responsive application title bar"""
        title_frame = tk.Frame(self.root, bg=self.colors['primary'], height=80)
        title_frame.pack(fill="x", padx=0, pady=0)
        title_frame.pack_propagate(False)
        
        # Responsive title container
        title_container = tk.Frame(title_frame, bg=self.colors['primary'])
        title_container.pack(expand=True, fill='both', padx=20, pady=10)
        
        # Main title
        title = tk.Label(title_container, 
                        text="UAV Power Line Inspection System",
                        font=self.get_font('title', 'bold'), 
                        bg=self.colors['primary'], 
                        fg="white")
        title.pack(expand=True)
        
        # Subtitle with status indicators
        subtitle_frame = tk.Frame(title_container, bg=self.colors['primary'])
        subtitle_frame.pack(expand=True, fill='x')
        
        subtitle = tk.Label(subtitle_frame,
                           text="AI-Powered Defect Detection • Real-Time 45-60 FPS • Automated Reporting",
                           font=self.get_font('subtitle'),
                           bg=self.colors['primary'],
                           fg="#bdc3c7")
        subtitle.pack(side='left')
        
        # System status indicator
        self.system_status = tk.Label(subtitle_frame,
                                    text="● READY",
                                    font=self.get_font('label', 'bold'),
                                    bg=self.colors['primary'],
                                    fg="#27ae60")
        self.system_status.pack(side='right')

    def create_left_panel(self, parent):
        """Create responsive left panel"""
        left_panel = tk.Frame(parent, width=300, bg="white", relief="flat", bd=1)
        left_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        left_panel.grid_propagate(False)
        
        # Make left panel scrollable for small screens
        self.setup_scrollable_frame(left_panel)
        
        # Detection Results Section
        self.create_detection_results_section(left_panel)
        
        # Performance Controls
        self.create_performance_controls(left_panel)
        
        # Control Buttons
        self.create_control_buttons(left_panel)
        
        # Database Controls
        self.create_database_controls(left_panel)
        
        # Statistics
        self.create_statistics_section(left_panel)

    def setup_scrollable_frame(self, parent):
        """Make panel scrollable for small screens"""
        # Create scrollable frame
        canvas = tk.Canvas(parent, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        self.scrollable_frame = tk.Frame(canvas, bg="white")
        
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw", width=280)
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack scrollable elements
        canvas.pack(side="left", fill="both", expand=True, padx=5)
        scrollbar.pack(side="right", fill="y")

    def create_detection_results_section(self, parent):
        """Create responsive detection results display"""
        results_header = tk.Label(self.scrollable_frame, text="🔍 DETECTION RESULTS", 
                                 font=self.get_font('header', 'bold'), 
                                 bg=self.colors['secondary'], 
                                 fg="white", 
                                 pady=10,
                                 anchor='w',
                                 padx=15)
        results_header.pack(fill="x", pady=(0, 8))
        
        # Current Detection Frame
        current_detection_frame = tk.Frame(self.scrollable_frame, bg="white", padx=10, pady=10)
        current_detection_frame.pack(fill="x", padx=5, pady=5)
        
        # Detection information variables
        self.detection_vars = {
            "label": tk.StringVar(value="No defects detected"),
            "confidence": tk.StringVar(value="0%"),
            "temperature": tk.StringVar(value="0.0 °C"),
            "timestamp": tk.StringVar(value="--:--:--"),
            "multiple_info": tk.StringVar(value="Single detection")
        }
        
        # Create detection cards
        self.create_detection_card(current_detection_frame, "Defect Type", "label", "#e74c3c")
        self.create_detection_card(current_detection_frame, "Confidence", "confidence", "#3498db")
        self.create_detection_card(current_detection_frame, "Temperature", "temperature", "#e67e22")
        self.create_detection_card(current_detection_frame, "Detection Info", "multiple_info", "#27ae60")
        self.create_detection_card(current_detection_frame, "Time Detected", "timestamp", "#2c3e50")

    def create_detection_card(self, parent, title, var_key, color):
        """Create a responsive detection info card"""
        card_frame = tk.Frame(parent, bg="#f8f9fa", relief="flat", bd=1, padx=8, pady=6)
        card_frame.pack(fill="x", pady=3)
        
        # Title
        title_label = tk.Label(card_frame, text=title, 
                              font=self.get_font('small'),
                              bg="#f8f9fa", fg="#7f8c8d",
                              anchor='w')
        title_label.pack(fill="x")
        
        # Value
        value_label = tk.Label(card_frame, textvariable=self.detection_vars[var_key],
                              font=self.get_font('label', 'bold'),
                              bg="#f8f9fa", fg=color,
                              anchor='w', wraplength=250)
        value_label.pack(fill="x")

    def create_performance_controls(self, parent):
        """Create responsive performance control section"""
        perf_frame = tk.LabelFrame(self.scrollable_frame, text="⚙️ PERFORMANCE SETTINGS", 
                                  font=self.get_font('header', 'bold'),
                                  bg="white", padx=12, pady=12,
                                  relief="flat", bd=1)
        perf_frame.pack(fill="x", padx=5, pady=8)
        
        # Detection Interval Control
        interval_frame = tk.Frame(perf_frame, bg="white")
        interval_frame.pack(fill="x", pady=6)
        
        tk.Label(interval_frame, text="Detection Interval:", 
                font=self.get_font('label'), bg="white").pack(side="left")
        
        self.interval_var = tk.StringVar(value="3")
        interval_spinbox = ttk.Spinbox(interval_frame, from_=1, to=10, width=6, 
                                      textvariable=self.interval_var,
                                      command=self.update_detection_interval)
        interval_spinbox.pack(side="right")
        
        # Processing toggle with better styling
        toggle_frame = tk.Frame(perf_frame, bg="white")
        toggle_frame.pack(fill="x", pady=6)
        
        self.processing_var = tk.BooleanVar(value=True)
        processing_cb = tk.Checkbutton(toggle_frame, text="Enable Real-time Detection", 
                                      variable=self.processing_var,
                                      command=self.toggle_processing,
                                      font=self.get_font('label'),
                                      bg="white",
                                      selectcolor="#e8f4fd")
        processing_cb.pack(side="left")
        
        # Performance info with status indicator
        self.perf_info_var = tk.StringVar(value="🎯 Smart Detection Active (Every 3rd frame)")
        perf_info = tk.Label(perf_frame, textvariable=self.perf_info_var,
                            font=self.get_font('small'), bg="#e8f4fd", fg="#2980b9",
                            padx=8, pady=4, relief="flat", wraplength=250)
        perf_info.pack(fill="x", pady=4)
        
        # Quick preset buttons
        preset_frame = tk.Frame(perf_frame, bg="white")
        preset_frame.pack(fill="x", pady=6)
        
        presets = [
            ("🔄 Balanced", "balanced", "#3498db"),
            ("⚡ Fast", "fast", "#27ae60"),
            ("🎯 Accurate", "accurate", "#e74c3c")
        ]
        
        for text, preset, color in presets:
            btn = tk.Button(preset_frame, text=text, font=self.get_font('tiny'),
                          command=lambda p=preset: self.set_performance_preset(p),
                          bg=color, fg="white", relief="flat",
                          padx=4, pady=2)
            btn.pack(side="left", expand=True, padx=1)

    def create_control_buttons(self, parent):
        """Create responsive control buttons section"""
        control_frame = tk.LabelFrame(self.scrollable_frame, text="🎮 INSPECTION CONTROLS", 
                                     font=self.get_font('header', 'bold'),
                                     bg="white", padx=12, pady=12,
                                     relief="flat", bd=1)
        control_frame.pack(fill="x", padx=5, pady=8)
        
        # Main control buttons
        self.start_btn = tk.Button(control_frame, text="🚀 START INSPECTION", 
                                  font=self.get_font('button_large', 'bold'),
                                  bg="#27ae60", fg="white",
                                  command=self.start_inspection,
                                  height=1, relief="flat",
                                  padx=10, pady=8)
        self.start_btn.pack(fill="x", pady=6)
        
        self.stop_btn = tk.Button(control_frame, text="🛑 STOP INSPECTION", 
                                 font=self.get_font('button_large', 'bold'),
                                 bg="#e74c3c", fg="white",
                                 command=self.stop_inspection,
                                 state="disabled",
                                 height=1, relief="flat",
                                 padx=10, pady=8)
        self.stop_btn.pack(fill="x", pady=6)
        
        # Quick actions frame
        quick_frame = tk.Frame(control_frame, bg="white")
        quick_frame.pack(fill="x", pady=6)
        
        quick_buttons = [
            ("📸 Snapshot", self.take_snapshot, "#f39c12"),
            ("🔄 Restart", self.restart_inspection, "#3498db"),
            ("⏸️ Pause", self.pause_inspection, "#9b59b6")
        ]
        
        for text, command, color in quick_buttons:
            btn = tk.Button(quick_frame, text=text, font=self.get_font('button_small'),
                          command=command, bg=color, fg="white",
                          relief="flat", padx=4, pady=4)
            btn.pack(side="left", expand=True, padx=2)

    def create_database_controls(self, parent):
        """Create responsive database control buttons"""
        db_frame = tk.LabelFrame(self.scrollable_frame, text="💾 DATA MANAGEMENT", 
                                font=self.get_font('header', 'bold'),
                                bg="white", padx=12, pady=12,
                                relief="flat", bd=1)
        db_frame.pack(fill="x", padx=5, pady=8)
        
        db_buttons = [
            ("📊 View Records", self.view_database_records, "#9b59b6"),
            ("🗑️ Clear Records", self.clear_database_records, "#e74c3c"),
            ("💾 Export CSV", self.export_data, "#3498db"),
            ("📄 Generate Report", self.generate_report, "#27ae60")
        ]
        
        for text, command, color in db_buttons:
            btn = tk.Button(db_frame, text=text, font=self.get_font('button_medium'),
                          command=command, bg=color, fg="white",
                          relief="flat", pady=6)
            btn.pack(fill="x", pady=3)

    def create_statistics_section(self, parent):
        """Create responsive statistics display section"""
        stats_frame = tk.LabelFrame(self.scrollable_frame, text="📈 LIVE STATISTICS", 
                                   font=self.get_font('header', 'bold'),
                                   bg="white", padx=12, pady=12,
                                   relief="flat", bd=1)
        stats_frame.pack(fill="x", padx=5, pady=8)
        
        self.stats_vars = {
            "total_detections": tk.StringVar(value="0"),
            "session_time": tk.StringVar(value="00:00:00"),
            "frame_rate": tk.StringVar(value="0.0 FPS"),
            "processing_rate": tk.StringVar(value="0.0 FPS"),
            "detection_interval": tk.StringVar(value="Interval: 3")
        }
        
        # Create stat cards
        stats_data = [
            ("Total Detections", "total_detections", "#e74c3c"),
            ("Session Duration", "session_time", "#3498db"),
            ("Camera FPS", "frame_rate", "#27ae60"),
            ("Processing FPS", "processing_rate", "#9b59b6"),
            ("Detection Mode", "detection_interval", "#f39c12")
        ]
        
        for title, var_key, color in stats_data:
            self.create_stat_card(stats_frame, title, var_key, color)

    def create_stat_card(self, parent, title, var_key, color):
        """Create a responsive statistics card"""
        card_frame = tk.Frame(parent, bg="#f8f9fa", relief="flat", bd=1, padx=10, pady=8)
        card_frame.pack(fill="x", pady=4)
        
        # Title
        title_label = tk.Label(card_frame, text=title, 
                              font=self.get_font('small'),
                              bg="#f8f9fa", fg="#7f8c8d",
                              anchor='w')
        title_label.pack(fill="x")
        
        # Value
        value_label = tk.Label(card_frame, textvariable=self.stats_vars[var_key],
                              font=self.get_font('label', 'bold'),
                              bg="#f8f9fa", fg=color,
                              anchor='w')
        value_label.pack(fill="x")

    def create_center_panel(self, parent):
        """Create responsive center panel with camera feed"""
        center_panel = tk.Frame(parent, bg="#2c3e50", relief="flat", bd=2)
        center_panel.grid(row=0, column=1, sticky="nsew", padx=5)
        
        # Configure center panel to expand
        parent.grid_columnconfigure(1, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        
        # Camera header
        camera_header = tk.Frame(center_panel, bg=self.colors['primary'], height=35)
        camera_header.pack(fill="x", pady=(0, 2))
        camera_header.pack_propagate(False)
        
        tk.Label(camera_header, text="📷 LIVE UAV CAMERA FEED", 
                font=self.get_font('header', 'bold'),
                bg=self.colors['primary'], fg="white").pack(expand=True)
        
        # Camera display area - responsive
        display_frame = tk.Frame(center_panel, bg="black")
        display_frame.pack(fill="both", expand=True)
        
        # Configure display frame to be responsive
        display_frame.grid_rowconfigure(0, weight=1)
        display_frame.grid_columnconfigure(0, weight=1)
        
        self.camera_display = tk.Label(display_frame, 
                                      text="UAV Camera Feed\n\nClick 'START INSPECTION' to begin\nreal-time power line inspection\n\n🎯 Optimized for 45-60 FPS\n🔍 AI-Powered Defect Detection",
                                      font=self.get_font('button_large'), 
                                      fg="white", bg="black",
                                      justify="center")
        self.camera_display.grid(row=0, column=0, sticky="nsew")
        
        # Camera info bar
        info_bar = tk.Frame(center_panel, bg="#34495e", height=25)
        info_bar.pack(fill="x", pady=(2, 0))
        info_bar.pack_propagate(False)
        
        self.camera_info = tk.StringVar(value="System Ready - Connect camera to begin")
        info_label = tk.Label(info_bar, textvariable=self.camera_info,
                             font=self.get_font('small'),
                             bg="#34495e", fg="#ecf0f1")
        info_label.pack(expand=True)

    def create_right_panel(self, parent):
        """Create responsive right panel with inspection details"""
        right_panel = tk.Frame(parent, width=280, bg="white", relief="flat", bd=1)
        right_panel.grid(row=0, column=2, sticky="nsew", padx=(10, 0))
        right_panel.grid_propagate(False)
        
        # Inspection Details Header
        details_header = tk.Label(right_panel, text="📋 INSPECTION DETAILS", 
                                 font=self.get_font('header', 'bold'), 
                                 bg=self.colors['secondary'], 
                                 fg="white", 
                                 pady=10,
                                 anchor='w',
                                 padx=15)
        details_header.pack(fill="x", pady=(0, 8))
        
        # Input Form with scrollbar
        self.create_input_form(right_panel)
        
        # Action Buttons
        self.create_action_buttons(right_panel)

    def create_input_form(self, parent):
        """Create responsive input form for inspection details"""
        # Create scrollable form container
        form_container = tk.Frame(parent, bg="white")
        form_container.pack(fill="both", expand=True, padx=12, pady=8)
        
        # Add scrollbar for many form fields
        canvas = tk.Canvas(form_container, bg="white", highlightthickness=0)
        scrollbar = ttk.Scrollbar(form_container, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="white")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        self.form_entries = {}
        
        # Use FORM_FIELDS from your config
        for field, unit in self.config.FORM_FIELDS:
            field_frame = tk.Frame(scrollable_frame, bg="white", pady=6)
            field_frame.pack(fill="x", pady=3)
            
            label_text = f"{field}:"
            if unit:
                label_text = f"{field} ({unit}):"
                
            tk.Label(field_frame, text=label_text, 
                    font=self.get_font('label'), 
                    bg="white", 
                    fg="#2c3e50",
                    anchor='w').pack(fill="x")
            
            entry = tk.Entry(field_frame, font=self.get_font('label'),
                           relief="flat", bg="#f8f9fa",
                           bd=1)
            entry.pack(fill="x", pady=(3, 0), ipady=6)
            self.form_entries[field] = entry
            
        # Pack scrollable elements
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Set default values for testing
        self.set_default_form_values()

    def create_action_buttons(self, parent):
        """Create responsive action buttons for right panel"""
        button_frame = tk.Frame(parent, bg="white", pady=12)
        button_frame.pack(fill="x", side="bottom", padx=12)
        
        # Main action buttons
        self.submit_btn = tk.Button(button_frame, text="💾 SAVE INSPECTION REPORT", 
                                   font=self.get_font('button_medium', 'bold'),
                                   bg="#3498db", fg="white",
                                   command=self.save_inspection_report,
                                   relief="flat", pady=10)
        self.submit_btn.pack(fill="x", pady=6)
        
        self.quick_save_btn = tk.Button(button_frame, text="⚡ QUICK SAVE", 
                                       font=self.get_font('button_small'),
                                       bg="#27ae60", fg="white",
                                       command=self.quick_save_inspection,
                                       relief="flat", pady=8)
        self.quick_save_btn.pack(fill="x", pady=3)
        
        # Form management buttons
        form_btn_frame = tk.Frame(button_frame, bg="white")
        form_btn_frame.pack(fill="x", pady=6)
        
        tk.Button(form_btn_frame, text="🧹 Clear Form", font=self.get_font('tiny'),
                 command=self.clear_form, bg="#95a5a6", fg="white",
                 relief="flat").pack(side="left", expand=True, padx=2)
        
        tk.Button(form_btn_frame, text="📝 Load Defaults", font=self.get_font('tiny'),
                 command=self.set_default_form_values, bg="#f39c12", fg="white",
                 relief="flat").pack(side="left", expand=True, padx=2)

    def create_status_bar(self):
        """Create responsive status bar at bottom"""
        status_frame = tk.Frame(self.root, bg="#34495e", height=28)
        status_frame.pack(side="bottom", fill="x")
        status_frame.pack_propagate(False)
        
        # Status message
        self.status_var = tk.StringVar(value="🔴 System Ready - Click 'Start Inspection' to begin")
        status_label = tk.Label(status_frame, textvariable=self.status_var, 
                               font=self.get_font('label'),
                               bg="#34495e", fg="#ecf0f1",
                               anchor='w', padx=15)
        status_label.pack(side="left", fill="x", expand=True)
        
        # System time
        self.time_var = tk.StringVar()
        time_label = tk.Label(status_frame, textvariable=self.time_var,
                             font=self.get_font('small'),
                             bg="#34495e", fg="#bdc3c7",
                             padx=15)
        time_label.pack(side="right")

    # ========== FIXED DATABASE METHODS ==========

    def set_default_form_values(self):
        """Set default values for form fields"""
        defaults = {
            "Distance from Target": "15",
            "Line Number": "LN-001",
            "Pole Number": "P-045",
            "Ambient Temperature": "25",
            "Weather Conditions": "Clear",
            "Inspector Name": "UAV Operator",
            "Wind Speed": "5",
            "Humidity": "45"
        }
        
        for field, value in defaults.items():
            if field in self.form_entries:
                self.form_entries[field].delete(0, tk.END)
                self.form_entries[field].insert(0, value)

    def clear_form(self):
        """Clear all form fields"""
        for entry in self.form_entries.values():
            entry.delete(0, tk.END)

    def set_performance_preset(self, preset):
        """Set performance presets"""
        presets = {
            "balanced": {"interval": 3, "processing": True},
            "fast": {"interval": 5, "processing": True},
            "accurate": {"interval": 1, "processing": True}
        }
        
        if preset in presets:
            config = presets[preset]
            self.interval_var.set(str(config["interval"]))
            self.processing_var.set(config["processing"])
            
            self.update_detection_interval()
            self.toggle_processing()
            
            self.status_var.set(f"🎯 {preset.capitalize()} mode activated")

    def update_detection_interval(self):
        """Update detection interval for performance"""
        try:
            interval = int(self.interval_var.get())
            self.detector.set_detection_interval(interval)
            self.stats_vars["detection_interval"].set(f"Interval: {interval}")
            self.perf_info_var.set(f"🎯 Smart Detection: Every {interval} frame(s)")
            self.status_var.set(f"🔧 Detection interval set to {interval}")
        except ValueError:
            self.interval_var.set("3")
            self.detector.set_detection_interval(3)

    def toggle_processing(self):
        """Toggle real-time processing"""
        enabled = self.processing_var.get()
        self.detector.set_processing_enabled(enabled)
        
        if enabled:
            interval = self.detector.get_detection_interval()
            self.status_var.set(f"🔄 Smart detection ENABLED (Interval: {interval})")
            self.perf_info_var.set(f"🎯 Smart Detection: Every {interval} frame(s)")
        else:
            self.status_var.set("⚪ Detection DISABLED - Smooth video only")
            self.perf_info_var.set("⚪ Detection Disabled - Max FPS")

    def update_time(self):
        """Update current time in status bar"""
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_var.set(current_time)
        self.root.after(1000, self.update_time)

    def pause_inspection(self):
        """Pause the inspection"""
        if self.running:
            self.running = False
            self.status_var.set("⏸️ Inspection Paused - Click Start to resume")
            self.system_status.config(text="● PAUSED", fg="#f39c12")
            messagebox.showinfo("Inspection Paused", "Inspection has been paused.")

    def start_inspection(self):
        """Start the inspection process using your existing CameraManager"""
        try:
            if not self.camera.start_capture():
                messagebox.showerror("Camera Error", 
                                   "Cannot access camera. Please check camera connection.")
                return
            
            self.running = True
            self.start_btn.config(state="disabled", bg="gray")
            self.stop_btn.config(state="normal", bg="#e74c3c")
            
            # Reset statistics
            self.detection_count = 0
            self.session_start_time = time.time()
            self.inspection_data = []
            
            self.status_var.set("🎥 Camera Active - Real-time inspection running...")
            self.system_status.config(text="● RUNNING", fg="#27ae60")
            self.camera_info.set("🔴 LIVE - Inspection in progress")
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
            
            # Start GUI updates
            self.update_gui()
            
            logging.info("✅ Inspection started successfully")
            
        except Exception as e:
            self.status_var.set(f"❌ Failed to start inspection: {str(e)}")
            messagebox.showerror("Startup Error", f"Cannot start inspection:\n{str(e)}")
            logging.error(f"Start inspection error: {e}")

    def stop_inspection(self):
        """Stop the inspection process using your existing CameraManager"""
        self.running = False
        self.camera.stop_capture()
        
        self.start_btn.config(state="normal", bg="#27ae60")
        self.stop_btn.config(state="disabled", bg="gray")
        
        self.status_var.set("🛑 Inspection Stopped - Ready for new session")
        self.system_status.config(text="● READY", fg="#27ae60")
        self.camera_info.set("System Ready - Connect camera to begin")
        logging.info("🛑 Inspection stopped")

    def restart_inspection(self):
        """Restart the inspection"""
        if self.running:
            self.stop_inspection()
            time.sleep(0.5)
        self.start_inspection()

    def take_snapshot(self):
        """Take a snapshot of the current frame"""
        if not self.running:
            messagebox.showwarning("No Active Inspection", "Please start inspection first")
            return
        
        try:
            if self.current_frame is not None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"snapshot_{timestamp}.jpg"
                cv2.imwrite(filename, self.current_frame)
                messagebox.showinfo("Snapshot Saved", f"Snapshot saved as {filename}")
                self.status_var.set(f"📸 Snapshot saved as {filename}")
            else:
                messagebox.showwarning("No Frame", "No camera frame available")
                
        except Exception as e:
            messagebox.showerror("Snapshot Error", f"Failed to take snapshot: {str(e)}")

    def _processing_loop(self):
        """Main processing loop using your existing PowerLineDetector"""
        while self.running:
            try:
                # Get frame from camera using your CameraManager
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.001)
                    continue
                
                # Store current frame for snapshot
                self.current_frame = frame.copy()
                
                # Process frame with detector using your PowerLineDetector
                processed_frame, detection_info = self.detector.process_frame(frame)
                
                # Update detection count
                if detection_info.get("label", "No Detection") != "No Detection" and detection_info.get("label") != "No Defects":
                    self.detection_count += 1
                    
                    # Save to database if processing is enabled and we have detections
                    if hasattr(self.detector, 'processing_enabled') and self.detector.processing_enabled:
                        form_data = self.get_form_data()
                        self.save_detection_to_database(detection_info, form_data)
                
                # Update detection display in main thread
                self.root.after(0, self._update_detection_display, processed_frame, detection_info)
                
            except Exception as e:
                logging.error(f"Processing loop error: {e}")
                time.sleep(0.01)

    def save_detection_to_database(self, detection_info, form_data):
        """Save detection to database with proper error handling"""
        try:
            # Prepare data for database
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Extract confidence value and convert to float if needed
            confidence_str = detection_info.get('confidence', '0%')
            confidence = 0.0
            if '%' in confidence_str:
                try:
                    confidence = float(confidence_str.replace('%', '')) / 100.0
                except ValueError:
                    confidence = 0.0
            
            # Extract temperature value
            temperature_str = detection_info.get('temperature', '0.0 °C')
            temperature = temperature_str.replace(' °C', '')
            
            # Prepare the data in the format expected by your database
            detection_data = {
                'timestamp': timestamp,
                'defect_type': detection_info.get('label', 'Unknown'),
                'confidence': confidence,
                'temperature': temperature,
                'line_number': form_data.get('Line Number', ''),
                'pole_number': form_data.get('Pole Number', ''),
                'inspector': form_data.get('Inspector Name', ''),
                'distance': form_data.get('Distance from Target', ''),
                'weather': form_data.get('Weather Conditions', ''),
                'wind_speed': form_data.get('Wind Speed', ''),
                'humidity': form_data.get('Humidity', '')
            }
            
            # Save using your database method
            if hasattr(self.database, 'save_detection'):
                success = self.database.save_detection(detection_data)
                if not success:
                    logging.warning("Failed to save detection to database")
            else:
                logging.warning("Database save_detection method not available")
                
        except Exception as e:
            logging.error(f"Error saving detection to database: {e}")

    def _update_detection_display(self, frame, detection_info):
        """Update detection display in main thread"""
        # Update detection information
        for key, value in detection_info.items():
            if key in self.detection_vars:
                self.detection_vars[key].set(str(value))
        
        # Update camera display
        self.display_annotated_frame(frame)

    def display_annotated_frame(self, frame):
        """Display the annotated frame in the GUI"""
        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            
            # Get current display size for responsive scaling
            display_width = self.camera_display.winfo_width()
            display_height = self.camera_display.winfo_height()
            
            # Use config dimensions if display size is small
            if display_width < 100 or display_height < 100:
                display_width = self.config.DISPLAY_WIDTH
                display_height = self.config.DISPLAY_HEIGHT
            
            img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
            
            imgtk = ImageTk.PhotoImage(image=img)
            
            self.camera_display.configure(image=imgtk, text="")
            self.camera_display.image = imgtk
            
        except Exception as e:
            logging.error(f"Frame display error: {e}")

    def update_gui(self):
        """Update GUI elements with improved performance monitoring"""
        if self.running:
            current_time = time.time()
            
            # Update FPS displays
            if current_time - self.last_gui_update >= 0.2:
                camera_fps = self.camera.get_fps()
                processing_fps = self.detector.get_processing_fps()
                
                self.stats_vars["frame_rate"].set(f"{camera_fps:.1f} FPS")
                self.stats_vars["processing_rate"].set(f"{processing_fps:.1f} FPS")
                self.stats_vars["total_detections"].set(str(self.detection_count))
                
                # Update performance indicators
                if camera_fps >= 50:
                    fps_color = "#27ae60"
                    perf_status = "Excellent"
                elif camera_fps >= 30:
                    fps_color = "#f39c12"
                    perf_status = "Good"
                else:
                    fps_color = "#e74c3c"
                    perf_status = "Poor"
                
                # Update status with performance info
                if hasattr(self.detector, 'processing_enabled') and self.detector.processing_enabled:
                    current_interval = self.detector.get_detection_interval()
                    status_text = f"🎥 Live Inspection • {perf_status} Performance • Detection: {processing_fps:.1f}FPS • Video: {camera_fps:.1f}FPS"
                    
                    self.system_status.config(fg=fps_color, text=f"● {perf_status.upper()}")
                else:
                    status_text = f"⚪ Display Mode • Video: {camera_fps:.1f} FPS • Detection Disabled"
                    self.system_status.config(fg="#3498db", text="● DISPLAY MODE")
                
                self.status_var.set(status_text)
                self.last_gui_update = current_time
            
            # Schedule next update
            self.root.after(50, self.update_gui)

    def update_session_timer(self):
        """Update session timer"""
        if self.running and self.session_start_time:
            elapsed = time.time() - self.session_start_time
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            self.stats_vars["session_time"].set(f"{hours:02d}:{minutes:02d}:{seconds:02d}")
        
        self.root.after(1000, self.update_session_timer)

    def get_form_data(self):
        """Get data from input form"""
        form_data = {}
        for field, entry in self.form_entries.items():
            form_data[field] = entry.get()
        return form_data

    def quick_save_inspection(self):
        """Quick save without dialog using your existing DatabaseManager"""
        try:
            form_data = self.get_form_data()
            
            # Validate required fields
            if not form_data.get('Line Number') or not form_data.get('Pole Number'):
                messagebox.showwarning("Missing Data", "Please enter Line Number and Pole Number")
                return
            
            # Get current detection info
            current_detection = getattr(self.detector, 'current_detections', {})
            
            # Save to database using your DatabaseManager
            success = self.save_manual_inspection_to_database(current_detection, form_data)
            if success:
                self.status_var.set("✅ Inspection data saved successfully!")
                messagebox.showinfo("Success", "Inspection data saved successfully!")
            else:
                self.status_var.set("❌ Failed to save inspection data")
                
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save inspection:\n{str(e)}")

    def save_manual_inspection_to_database(self, detection_info, form_data):
        """Save manual inspection to database with proper data formatting"""
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Extract confidence value
            confidence_str = detection_info.get('confidence', '0%')
            confidence = 0.0
            if '%' in confidence_str:
                try:
                    confidence = float(confidence_str.replace('%', '')) / 100.0
                except ValueError:
                    confidence = 0.0
            
            # Extract temperature value
            temperature_str = detection_info.get('temperature', '0.0 °C')
            temperature = temperature_str.replace(' °C', '')
            
            # Prepare inspection data
            inspection_data = {
                'timestamp': timestamp,
                'defect_type': detection_info.get('label', 'Manual Inspection'),
                'confidence': confidence,
                'temperature': temperature,
                'line_number': form_data.get('Line Number', ''),
                'pole_number': form_data.get('Pole Number', ''),
                'inspector': form_data.get('Inspector Name', ''),
                'distance': form_data.get('Distance from Target', ''),
                'weather': form_data.get('Weather Conditions', ''),
                'wind_speed': form_data.get('Wind Speed', ''),
                'humidity': form_data.get('Humidity', ''),
                'ambient_temperature': form_data.get('Ambient Temperature', '')
            }
            
            # Use the appropriate database method
            if hasattr(self.database, 'save_manual_inspection'):
                return self.database.save_manual_inspection(inspection_data)
            elif hasattr(self.database, 'save_detection'):
                return self.database.save_detection(inspection_data)
            else:
                logging.warning("No suitable database save method found")
                return False
                
        except Exception as e:
            logging.error(f"Error saving manual inspection: {e}")
            return False

    def save_inspection_report(self):
        """Save inspection report with current data using your existing DatabaseManager"""
        try:
            form_data = self.get_form_data()
            
            # Validate required fields
            if not form_data.get('Line Number') or not form_data.get('Pole Number'):
                messagebox.showwarning("Missing Data", "Please enter Line Number and Pole Number")
                return
            
            # Get current detection info or create default
            current_detection = getattr(self.detector, 'current_detections', {})
            if current_detection.get("label") != "No Detection":
                detection_info = current_detection
            else:
                detection_info = {
                    "label": "Manual Inspection - No Defects",
                    "confidence": "0%",
                    "temperature": form_data.get('Ambient Temperature', '0.0') + ' °C',
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "multiple_info": ""
                }
            
            # Save to database using your DatabaseManager
            success = self.save_manual_inspection_to_database(detection_info, form_data)
            if success:
                messagebox.showinfo("Success", "Inspection report saved to database successfully!")
                self.status_var.set("✅ Inspection report saved to database")
                
                # Generate text report
                self.generate_text_report(detection_info, form_data)
            else:
                messagebox.showerror("Error", "Failed to save inspection report to database")
                
        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save report:\n{str(e)}")

    def generate_text_report(self, detection_info, form_data):
        """Generate a text inspection report"""
        try:
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
                
                f.write(f"\nDetection Results:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Defect Type: {detection_info['label']}\n")
                f.write(f"Confidence: {detection_info['confidence']}\n")
                f.write(f"Temperature: {detection_info['temperature']}\n")
                f.write(f"Timestamp: {detection_info['timestamp']}\n")
                
                if detection_info['multiple_info']:
                    f.write(f"\nMultiple Detections:\n{detection_info['multiple_info']}\n")
                
                f.write(f"\nSession Statistics:\n")
                f.write("-" * 20 + "\n")
                f.write(f"Total Detections: {self.detection_count}\n")
                f.write(f"Session Duration: {self.stats_vars['session_time'].get()}\n")
            
            messagebox.showinfo("Report Generated", f"Text report saved as:\n{filename}")
            self.status_var.set(f"📄 Text report generated: {filename}")
            
        except Exception as e:
            messagebox.showerror("Report Error", f"Failed to generate text report:\n{str(e)}")

    def generate_report(self):
        """Generate comprehensive inspection report"""
        try:
            form_data = self.get_form_data()
            detection_info = getattr(self.detector, 'current_detections', {})
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_report_{timestamp}.txt"
            
            with open(filename, 'w') as f:
                f.write("=" * 60 + "\n")
                f.write("        UAV POWER LINE INSPECTION REPORT\n")
                f.write("=" * 60 + "\n\n")
                
                f.write("SESSION SUMMARY:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Report Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Session Duration: {self.stats_vars['session_time'].get()}\n")
                f.write(f"Total Detections: {self.detection_count}\n")
                f.write(f"Processing FPS: {self.stats_vars['processing_rate'].get()}\n\n")
                
                f.write("INSPECTION DETAILS:\n")
                f.write("-" * 40 + "\n")
                for field, value in form_data.items():
                    f.write(f"{field}: {value}\n")
                
                f.write(f"\nDETECTION RESULTS:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Defect Type: {detection_info.get('label', 'No defects')}\n")
                f.write(f"Confidence: {detection_info.get('confidence', '0%')}\n")
                f.write(f"Temperature: {detection_info.get('temperature', 'N/A')}\n")
                f.write(f"Timestamp: {detection_info.get('timestamp', 'N/A')}\n")
                
                f.write(f"\nSYSTEM INFORMATION:\n")
                f.write("-" * 40 + "\n")
                f.write(f"Detection Interval: {self.interval_var.get()}\n")
                f.write(f"Real-time Processing: {'Enabled' if self.processing_var.get() else 'Disabled'}\n")
                f.write(f"Camera FPS: {self.stats_vars['frame_rate'].get()}\n")
            
            messagebox.showinfo("Report Generated", f"Comprehensive inspection report saved as:\n{filename}")
            self.status_var.set(f"📄 Report generated: {filename}")
            
        except Exception as e:
            messagebox.showerror("Report Error", f"Failed to generate report:\n{str(e)}")

    def export_data(self):
        """Export data to CSV using your existing DatabaseManager"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"inspection_data_{timestamp}.csv"
            
            if hasattr(self.database, 'export_to_csv'):
                success, message = self.database.export_to_csv(filename)
                if success:
                    messagebox.showinfo("Data Exported", message)
                    self.status_var.set(f"💾 {message}")
                else:
                    messagebox.showerror("Export Error", message)
            else:
                messagebox.showerror("Export Error", "Database export method not available")
                
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export data:\n{str(e)}")

    def view_database_records(self):
        """Display database records in a new window using your existing DatabaseManager"""
        try:
            if hasattr(self.database, 'get_recent_records'):
                records = self.database.get_recent_records(100)
            else:
                records = []
            
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
            
            if not records:
                no_data_label = tk.Label(db_window, 
                                       text="No records available in database",
                                       font=("Arial", 12),
                                       bg="white", fg="gray")
                no_data_label.pack(expand=True)
                return
            
            # Create treeview
            tree_frame = tk.Frame(db_window)
            tree_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            columns = ("Timestamp", "Defect Type", "Confidence", "Temperature", "Line Number", "Pole Number", "Inspector")
            tree = ttk.Treeview(tree_frame, columns=columns, show="headings", height=20)
            
            # Define headings and columns
            for col in columns:
                tree.heading(col, text=col)
                tree.column(col, width=120, anchor="center")
            
            tree.column("Defect Type", width=200, anchor="w")
            tree.column("Timestamp", width=150)
            
            # Add scrollbars
            v_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
            h_scrollbar = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
            tree.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)
            
            # Pack treeview and scrollbars
            tree.grid(row=0, column=0, sticky="nsew")
            v_scrollbar.grid(row=0, column=1, sticky="ns")
            h_scrollbar.grid(row=1, column=0, sticky="ew")
            
            tree_frame.grid_rowconfigure(0, weight=1)
            tree_frame.grid_columnconfigure(0, weight=1)
            
            # Insert records
            for record in records:
                # Handle different record formats from your database
                if hasattr(record, '__getitem__'):
                    confidence = record[2] if len(record) > 2 else '0%'
                    if isinstance(confidence, (int, float)):
                        confidence = f"{confidence * 100:.1f}%"
                    elif isinstance(confidence, str) and '%' not in confidence:
                        try:
                            confidence_val = float(confidence) * 100
                            confidence = f"{confidence_val:.1f}%"
                        except ValueError:
                            confidence = "0%"
                    
                    formatted_record = (
                        record[0] if len(record) > 0 else 'N/A',
                        record[1] if len(record) > 1 else 'N/A',
                        confidence,
                        record[3] if len(record) > 3 else 'N/A',
                        record[4] if len(record) > 4 else 'N/A',
                        record[5] if len(record) > 5 else 'N/A',
                        record[6] if len(record) > 6 else 'N/A'
                    )
                    tree.insert("", "end", values=formatted_record)
            
            # Add record count
            count_label = tk.Label(db_window, 
                                  text=f"Total records displayed: {len(records)}",
                                  font=("Arial", 10, "bold"),
                                  bg="white", fg="navy")
            count_label.pack(pady=5)
            
        except Exception as e:
            messagebox.showerror("Database Error", f"Failed to read records: {str(e)}")

    def clear_database_records(self):
        """Clear all inspection records using your existing DatabaseManager"""
        if messagebox.askyesno("Confirm Clear", 
                              "Are you sure you want to delete ALL inspection records?\nThis action cannot be undone."):
            if hasattr(self.database, 'clear_records'):
                success = self.database.clear_records()
                if success:
                    messagebox.showinfo("Success", "All inspection records have been cleared.")
                    self.status_var.set("🗑️ All inspection records cleared")
                else:
                    messagebox.showerror("Error", "Failed to clear records")
            else:
                messagebox.showerror("Error", "Database clear method not available")

    def on_closing(self):
        """Handle application closing with confirmation"""
        if messagebox.askyesno("Confirm Exit", 
                              "Are you sure you want to exit?\n\nAny unsaved data will be lost."):
            self.running = False
            self.camera.stop_capture()
            if hasattr(self.database, 'close'):
                self.database.close()
            self.root.destroy()
            logging.info("🔴 Application closed safely")

# Application entry point
def main():
    """Main application entry point"""
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        root = tk.Tk()
        app = PowerLineInspectorGUI(root)
        
        # Center the window on screen
        root.update_idletasks()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = (screen_width // 2) - (app.config.WINDOW_WIDTH // 2)
        y = (screen_height // 2) - (app.config.WINDOW_HEIGHT // 2)
        root.geometry(f"+{x}+{y}")

        # Handle window close event
        root.protocol("WM_DELETE_WINDOW", app.on_closing)

        root.mainloop()
        
    except Exception as e:
        logging.error(f"❌ Application failed to start: {e}")
        messagebox.showerror("Startup Error", 
                           f"Application failed to start:\n{str(e)}\n\n"
                           "Please check:\n"
                           "• Camera connection\n"
                           "• Required dependencies")

if __name__ == "__main__":
    main()