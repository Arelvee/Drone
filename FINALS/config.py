import os
import cv2

class Config:
    """Configuration settings for the Power Line Inspection System - FINAL OPTIMIZED"""
    
    # ==================== MODEL SETTINGS ====================
    MODEL_PATH = 'finalsirbagsic.pt'
    CONFIDENCE_THRESHOLD = 0.10  # Balanced threshold for accuracy vs performance
    IMAGE_SIZE = 480  # Optimized for speed while maintaining detection quality
    
    # ==================== CAMERA SETTINGS ====================
    CAMERA_WIDTH = 640   # Optimal resolution for performance
    CAMERA_HEIGHT = 480  # 4:3 aspect ratio
    CAMERA_FPS = 30      # Realistic camera FPS
    CAMERA_BACKENDS = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2, cv2.CAP_ANY]
    VIDEO_PATH = "TEST2-Drone.mp4"
    
    # ==================== PROCESSING SETTINGS ====================
    TARGET_FPS = 30      # Realistic target for smooth video with processing
    MIN_FPS = 20         # Minimum acceptable FPS
    FRAME_QUEUE_SIZE = 2 # Balanced queue size
    MAX_FRAME_SKIP = 2   # Reasonable skipping for performance
    
    # ==================== GUI SETTINGS ====================
    WINDOW_WIDTH = 1200
    WINDOW_HEIGHT = 700
    DISPLAY_WIDTH = 640
    DISPLAY_HEIGHT = 480
    
    # ==================== DATABASE SETTINGS ====================
    DATABASE_PATH = 'power_line_inspection.db'
    
    # ==================== CLASS DEFINITIONS ====================
    CLASS_NAMES = {
        0: "1-1A-Fired Wedge Joint-BARE",
        1: "1-1B-Fired Wedge Joint-COVERED", 
        2: "2-2A-Hummer Driven Wedge Joint-BARE",
        3: "2-2B-Hummer Driven Wedge Joint-COVERED"
    }
    
    # ==================== TEMPERATURE SIMULATION ====================
    TEMPERATURE_RANGES = {
        0: (35.0, 65.0),   # 1-1A-Fired Wedge Joint-BARE (hottest)
        1: (30.0, 55.0),   # 1-1B-Fired Wedge Joint-COVERED
        2: (32.0, 60.0),   # 2-2A-Hummer Driven Wedge Joint-BARE
        3: (28.0, 50.0)    # 2-2B-Hummer Driven Wedge Joint-COVERED (coolest)
    }
    
    # ==================== PERFORMANCE OPTIMIZATION SETTINGS ====================
    USE_GPU = True                    # Use GPU acceleration if available
    DETECTION_INTERVAL = 3            # Process every 3rd frame (balanced performance)
    MOTION_THRESHOLD = 1500           # Motion sensitivity for smart processing
    BUFFER_DETECTIONS = True          # Smooth display by reusing recent detections
    MAX_DETECTION_AGE = 8             # Frames to keep detection active
    ADAPTIVE_PROCESSING = True        # Automatically adjust based on system load
    
    # ==================== VIDEO PLAYBACK SETTINGS ====================
    VIDEO_PLAYBACK_MODE = "smooth"    # "smooth" or "realtime"
    MAX_VIDEO_FPS = 30                # Cap video playback FPS
    SYNC_TO_MONITOR = True            # Sync with monitor refresh rate
    
    # ==================== DETECTION OPTIMIZATION ====================
    ENABLE_TRACKING = False           # Object tracking (can be CPU intensive)
    MAX_CONCURRENT_DETECTIONS = 6     # Limit simultaneous detections
    DETECTION_TIMEOUT = 0.5           # Max time allowed per detection
    
    # ==================== QUALITY SETTINGS ====================
    DRAW_SIMPLE_BOXES = True          # Use simple bounding boxes for performance
    ENABLE_TEMPERATURE_SIMULATION = True
    SHOW_CONFIDENCE_SCORES = True
    DISPLAY_MULTIPLE_DETECTIONS = True
    
    # ==================== FORM FIELDS ====================
    FORM_FIELDS = [
        ("Distance from Target", "meters"),
        ("Line Number", ""),
        ("Pole Number", ""),
        ("Ambient Temperature", "°C"),
        ("Weather Conditions", ""),
        ("Inspector Name", "")
    ]
    
    # ==================== PERFORMANCE PRESETS ====================
    PERFORMANCE_PRESETS = {
        "fast": {
            "detection_interval": 5,
            "image_size": 640,
            "confidence_threshold": 0.35,
            "draw_simple_boxes": False
        },
        "balanced": {
            "detection_interval": 2,
            "image_size": 480,
            "confidence_threshold": 0.25,
            "draw_simple_boxes": False
        },
        "accurate": {
            "detection_interval": 1,
            "image_size": 640,
            "confidence_threshold": 0.15,
            "draw_simple_boxes": False
        }
    }
    
    # ==================== VALIDATION METHODS ====================
    
    @classmethod
    def validate_settings(cls):
        """Validate all configuration settings"""
        # Check model file exists
        if not os.path.exists(cls.MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {cls.MODEL_PATH}")
        
        # Check video file exists (if not using camera)
        if cls.VIDEO_PATH and cls.VIDEO_PATH != "0" and not os.path.exists(cls.VIDEO_PATH):
            print(f"⚠️  Video file not found: {cls.VIDEO_PATH} - Will try camera instead")
        
        # Validate ranges
        if not (0 <= cls.CONFIDENCE_THRESHOLD <= 1):
            raise ValueError("Confidence threshold must be between 0 and 1")
        
        if cls.DETECTION_INTERVAL < 1:
            raise ValueError("Detection interval must be at least 1")
        
        if cls.TARGET_FPS < 1 or cls.TARGET_FPS > 120:
            raise ValueError("Target FPS must be between 1 and 120")
        
        print("✅ Configuration validated successfully")
    
    @classmethod
    def get_preset(cls, preset_name):
        """Get performance preset configuration"""
        if preset_name in cls.PERFORMANCE_PRESETS:
            return cls.PERFORMANCE_PRESETS[preset_name]
        else:
            print(f"⚠️  Unknown preset: {preset_name}. Using 'balanced'.")
            return cls.PERFORMANCE_PRESETS["balanced"]
    
    @classmethod
    def apply_preset(cls, preset_name):
        """Apply a performance preset"""
        preset = cls.get_preset(preset_name)
        
        cls.DETECTION_INTERVAL = preset["detection_interval"]
        cls.IMAGE_SIZE = preset["image_size"]
        cls.CONFIDENCE_THRESHOLD = preset["confidence_threshold"]
        cls.DRAW_SIMPLE_BOXES = preset["draw_simple_boxes"]
        
        print(f"🎯 Applied '{preset_name}' performance preset")
    
    @classmethod
    def print_current_settings(cls):
        """Print current configuration settings"""
        print("\n" + "="*50)
        print("CURRENT CONFIGURATION SETTINGS")
        print("="*50)
        
        settings_groups = {
            "Model Settings": {
                "MODEL_PATH": cls.MODEL_PATH,
                "CONFIDENCE_THRESHOLD": cls.CONFIDENCE_THRESHOLD,
                "IMAGE_SIZE": cls.IMAGE_SIZE
            },
            "Camera Settings": {
                "CAMERA_WIDTH": cls.CAMERA_WIDTH,
                "CAMERA_HEIGHT": cls.CAMERA_HEIGHT,
                "CAMERA_FPS": cls.CAMERA_FPS,
                "VIDEO_PATH": cls.VIDEO_PATH
            },
            "Performance Settings": {
                "TARGET_FPS": cls.TARGET_FPS,
                "DETECTION_INTERVAL": cls.DETECTION_INTERVAL,
                "USE_GPU": cls.USE_GPU,
                "BUFFER_DETECTIONS": cls.BUFFER_DETECTIONS
            },
            "Optimization Settings": {
                "ADAPTIVE_PROCESSING": cls.ADAPTIVE_PROCESSING,
                "MOTION_THRESHOLD": cls.MOTION_THRESHOLD,
                "MAX_CONCURRENT_DETECTIONS": cls.MAX_CONCURRENT_DETECTIONS
            }
        }
        
        for group_name, settings in settings_groups.items():
            print(f"\n{group_name}:")
            for key, value in settings.items():
                print(f"  {key}: {value}")
        
        print("\n" + "="*50)

# Auto-validate settings when module is imported
try:
    Config.validate_settings()
    Config.print_current_settings()
except Exception as e:
    print(f"❌ Configuration error: {e}")