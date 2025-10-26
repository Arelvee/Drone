import sqlite3
import logging
import csv
from datetime import datetime
from config import Config

class DatabaseManager:
    """Manages database operations for inspection records"""
    
    def __init__(self):
        self.config = Config()
        self.conn = None
        self.cursor = None
        self.initialize_database()
    
    def initialize_database(self):
        """Initialize SQLite database with required tables"""
        try:
            self.conn = sqlite3.connect(self.config.DATABASE_PATH, check_same_thread=False)
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
            
            # Create class labels table
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS class_labels (
                    id INTEGER PRIMARY KEY,
                    class_name TEXT NOT NULL UNIQUE,
                    description TEXT
                )
            ''')
            
            # Insert predefined class names
            for class_id, class_name in self.config.CLASS_NAMES.items():
                self.cursor.execute('''
                    INSERT OR IGNORE INTO class_labels (id, class_name, description)
                    VALUES (?, ?, ?)
                ''', (class_id, class_name, f"Power line joint type: {class_name}"))
            
            self.conn.commit()
            logging.info("✅ Database initialized successfully")
            
        except sqlite3.Error as e:
            logging.error(f"Database initialization error: {e}")
            raise
    
    def save_detection(self, detection_info, form_data):
        """Save detection information to database"""
        try:
            if detection_info.get("label", "No Detection") != "No Detection" and "all_detections" in detection_info:
                # Save each detection individually
                for label, confidence, class_id, temperature in detection_info["all_detections"]:
                    self.cursor.execute('''
                        INSERT INTO inspection_records 
                        (timestamp, defect_type, confidence, temperature, distance, 
                         line_number, pole_number, ambient_temperature, weather_conditions, inspector_name)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        detection_info['timestamp'],
                        label,
                        confidence,
                        f"{temperature:.1f} °C",
                        form_data.get('Distance from Target', ''),
                        form_data.get('Line Number', ''),
                        form_data.get('Pole Number', ''),
                        form_data.get('Ambient Temperature', ''),
                        form_data.get('Weather Conditions', ''),
                        form_data.get('Inspector Name', '')
                    ))
                    
                self.conn.commit()
                logging.info(f"✅ {len(detection_info['all_detections'])} detections saved to database")
                return True
            return False
                
        except sqlite3.Error as e:
            logging.error(f"Database save error: {e}")
            return False
    
    def save_manual_inspection(self, detection_info, form_data):
        """Save manual inspection record"""
        try:
            self.cursor.execute('''
                INSERT INTO inspection_records 
                (timestamp, defect_type, confidence, temperature, distance, 
                 line_number, pole_number, ambient_temperature, weather_conditions, inspector_name)
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
            logging.info("✅ Manual inspection saved to database")
            return True
            
        except sqlite3.Error as e:
            logging.error(f"Manual inspection save error: {e}")
            return False
    
    def get_recent_records(self, limit=100):
        """Get recent inspection records"""
        try:
            self.cursor.execute('''
                SELECT timestamp, defect_type, confidence, temperature, 
                       line_number, pole_number, inspector_name
                FROM inspection_records 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            logging.error(f"Database read error: {e}")
            return []
    
    def get_all_records(self):
        """Get all inspection records"""
        try:
            self.cursor.execute('''
                SELECT timestamp, defect_type, confidence, temperature, 
                       distance, line_number, pole_number, ambient_temperature, 
                       weather_conditions, inspector_name
                FROM inspection_records
                ORDER BY timestamp
            ''')
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            logging.error(f"Database read error: {e}")
            return []
    
    def clear_records(self):
        """Clear all inspection records"""
        try:
            self.cursor.execute("DELETE FROM inspection_records")
            self.conn.commit()
            logging.info("🗑️ All inspection records cleared")
            return True
        except sqlite3.Error as e:
            logging.error(f"Database clear error: {e}")
            return False
    
    def export_to_csv(self, filename):
        """Export records to CSV file"""
        try:
            records = self.get_all_records()
            if not records:
                return False, "No records to export"
            
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "DefectType", "Confidence", "Temperature", "Distance", 
                               "LineNumber", "PoleNumber", "AmbientTemperature", "WeatherConditions", "InspectorName"])
                
                for record in records:
                    writer.writerow(record)
            
            return True, f"Exported {len(records)} records to {filename}"
            
        except Exception as e:
            return False, f"Export error: {e}"
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logging.info("🔴 Database connection closed")