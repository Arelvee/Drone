# image_detector.py
import cv2
import os
import argparse
from ultralytics import YOLO
import numpy as np
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ImageDetector:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.class_names = {
            0: "1-1A-Fired Wedge Joint-BARE",
            1: "1-1B-Fired Wedge Joint-COVERED", 
            2: "2-2A-Hummer Driven Wedge Joint-BARE",
            3: "2-2B-Hummer Driven Wedge Joint-COVERED"
        }
        self.load_model()
    
    def load_model(self):
        """Load YOLO model"""
        if not os.path.exists(self.model_path):
            logging.error(f"Model file '{self.model_path}' not found!")
            return False
        
        try:
            self.model = YOLO(self.model_path)
            logging.info(f"✅ Loaded model from {self.model_path}")
            
            # Update model class names with our specific classes
            if hasattr(self.model, 'names'):
                for i, name in self.class_names.items():
                    if i < len(self.model.names):
                        self.model.names[i] = name
            return True
            
        except Exception as e:
            logging.error(f"❌ Error loading model: {e}")
            return False
    
    def calculate_temperature(self, confidence):
        """Calculate temperature based on confidence"""
        base_temp = 25.0
        temp_increase = confidence * 15  # Higher confidence = higher temperature
        return base_temp + temp_increase
    
    def detect_image(self, image_path, output_dir="output_detections", confidence_threshold=0.1):
        """Detect objects in a single image"""
        if not self.model:
            logging.error("Model not loaded!")
            return False
        
        if not os.path.exists(image_path):
            logging.error(f"Image file '{image_path}' not found!")
            return False
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                logging.error(f"Failed to read image: {image_path}")
                return False
            
            original_image = image.copy()
            
            # Run YOLO inference
            results = self.model(image, conf=confidence_threshold, verbose=False)
            
            if len(results[0].boxes) == 0:
                logging.info("❌ No detections found in the image")
                # Save original image even if no detections
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(output_dir, f"no_detection_{timestamp}.jpg")
                cv2.imwrite(output_path, original_image)
                return False
            
            # Process detections
            boxes = results[0].boxes
            detections_count = len(boxes)
            
            logging.info(f"🎯 Found {detections_count} detection(s) in the image")
            
            # Process all detections and add annotations
            for i in range(len(boxes)):
                confidence = float(boxes.conf[i])
                class_id = int(boxes.cls[i])
                
                # Get class name
                class_name = self.class_names.get(class_id, f"Unknown Class {class_id}")
                
                # Calculate temperature for this detection
                temperature = self.calculate_temperature(confidence)
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw bounding box
                color = self.get_color(class_id)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Create label with class, confidence, and temperature
                label = f"{class_name}: {confidence:.1%} - {temperature:.1f}°C"
                
                # Calculate text background
                (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Draw text background
                cv2.rectangle(image, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
                
                # Put text
                cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                logging.info(f"  🔍 Detection {i+1}: {class_name} - Confidence: {confidence:.1%} - Temperature: {temperature:.1f}°C")
            
            # Add summary information on the image
            self.add_summary_info(image, detections_count)
            
            # Save the annotated image
            original_name = os.path.splitext(os.path.basename(image_path))[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"detected_{original_name}_{timestamp}.jpg")
            
            cv2.imwrite(output_path, image)
            logging.info(f"💾 Saved annotated image to: {output_path}")
            
            # Also save a comparison image (side by side)
            self.save_comparison_image(original_image, image, original_name, output_dir, timestamp)
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Error processing image: {e}")
            return False
    
    def detect_images_in_folder(self, folder_path, output_dir="output_detections", confidence_threshold=0.1):
        """Detect objects in all images in a folder"""
        if not self.model:
            logging.error("Model not loaded!")
            return False
        
        if not os.path.exists(folder_path):
            logging.error(f"Folder '{folder_path}' not found!")
            return False
        
        # Supported image extensions
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        # Get all image files in the folder
        image_files = []
        for file in os.listdir(folder_path):
            if os.path.splitext(file)[1].lower() in valid_extensions:
                image_files.append(os.path.join(folder_path, file))
        
        if not image_files:
            logging.error(f"No image files found in '{folder_path}'")
            return False
        
        logging.info(f"📁 Found {len(image_files)} image(s) to process")
        
        success_count = 0
        for image_path in image_files:
            logging.info(f"\n🔍 Processing: {os.path.basename(image_path)}")
            if self.detect_image(image_path, output_dir, confidence_threshold):
                success_count += 1
        
        logging.info(f"\n📊 Processing complete: {success_count}/{len(image_files)} images successfully processed")
        return success_count > 0
    
    def get_color(self, class_id):
        """Get unique color for each class"""
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue  
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        return colors[class_id % len(colors)]
    
    def add_summary_info(self, image, detections_count):
        """Add summary information to the image"""
        height, width = image.shape[:2]
        
        # Create summary text
        summary_text = [
            f"Total Detections: {detections_count}",
            f"Processing Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "Powered by YOLO Power Line Inspector"
        ]
        
        # Add summary background
        cv2.rectangle(image, (10, height - 100), (width - 10, height - 10), (0, 0, 0), -1)
        
        # Add summary text
        y_position = height - 75
        for text in summary_text:
            cv2.putText(image, text, (20, y_position), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (255, 255, 255), 1)
            y_position += 20
    
    def save_comparison_image(self, original, annotated, original_name, output_dir, timestamp):
        """Save a side-by-side comparison image"""
        # Resize images to same height for comparison
        height = max(original.shape[0], annotated.shape[0])
        width1 = int(original.shape[1] * (height / original.shape[0]))
        width2 = int(annotated.shape[1] * (height / annotated.shape[0]))
        
        original_resized = cv2.resize(original, (width1, height))
        annotated_resized = cv2.resize(annotated, (width2, height))
        
        # Create comparison image
        comparison = np.hstack([original_resized, annotated_resized])
        
        # Add divider line
        cv2.line(comparison, (width1, 0), (width1, height), (255, 255, 255), 2)
        
        # Add labels
        cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(comparison, "Detected", (width1 + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Save comparison image
        comparison_path = os.path.join(output_dir, f"comparison_{original_name}_{timestamp}.jpg")
        cv2.imwrite(comparison_path, comparison)
        logging.info(f"📊 Saved comparison image to: {comparison_path}")

def main():
    parser = argparse.ArgumentParser(description='Power Line Joint Detector - Image Mode')
    parser.add_argument('--model', type=str, default='finalsirbagsic.pt', 
                       help='Path to YOLO model file (default: finalsirbagsic.pt)')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image or folder containing images')
    parser.add_argument('--output', type=str, default='output_detections',
                       help='Output directory for detected images (default: output_detections)')
    parser.add_argument('--confidence', type=float, default=0.1,
                       help='Confidence threshold for detection (default: 0.1)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = ImageDetector(args.model)
    
    if not detector.model:
        return
    
    # Check if input is a file or folder
    if os.path.isfile(args.input):
        # Single image detection
        detector.detect_image(args.input, args.output, args.confidence)
    elif os.path.isdir(args.input):
        # Folder of images detection
        detector.detect_images_in_folder(args.input, args.output, args.confidence)
    else:
        logging.error(f"Input path '{args.input}' does not exist!")

if __name__ == "__main__":
    main()