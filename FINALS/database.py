import sqlite3
import logging
import csv
from datetime import datetime
from config import Config

class DatabaseManager:
    """Manages database operations for power line inspection records"""
    
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
                    joint_type TEXT NOT NULL,
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
            
            self.conn.commit()
            logging.info("✅ Database initialized successfully")
            
        except sqlite3.Error as e:
            logging.error(f"❌ Database initialization error: {e}")
            raise
    
    def save_inspection(self, inspection_data, form_data=None):
        """
        Unified method to save inspection data
        
        Args:
            inspection_data (dict): Detection information containing:
                - label: Joint type/defect type
                - confidence: Confidence percentage as string or float
                - temperature: Temperature reading as string
            form_data (dict, optional): Form data containing inspection details
        """
        try:
            # Get current timestamp
            current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Extract form data
            if form_data:
                distance = form_data.get('Distance from Target', '')
                line_number = form_data.get('Line Number', '')
                pole_number = form_data.get('Pole Number', '')
                ambient_temperature = form_data.get('Ambient Temperature', '')
                weather_conditions = form_data.get('Weather Conditions', '')
                inspector_name = form_data.get('Inspector Name', '')
            else:
                # Extract from inspection_data if form_data is not provided
                distance = inspection_data.get('distance', '')
                line_number = inspection_data.get('line_number', '')
                pole_number = inspection_data.get('pole_number', '')
                ambient_temperature = inspection_data.get('ambient_temperature', '')
                weather_conditions = inspection_data.get('weather_conditions', inspection_data.get('weather', ''))
                inspector_name = inspection_data.get('inspector_name', inspection_data.get('inspector', ''))
            
            # Extract and convert confidence value
            confidence_str = str(inspection_data.get('confidence', '0%'))
            confidence = 0.0
            try:
                if '%' in confidence_str:
                    confidence = float(confidence_str.replace('%', '')) / 100.0
                else:
                    confidence = float(confidence_str)
            except (ValueError, TypeError):
                confidence = 0.0
                logging.warning(f"⚠️ Could not parse confidence value: {confidence_str}")
            
            # Extract temperature value
            temperature_str = str(inspection_data.get('temperature', '0.0 °C'))
            
            # Extract joint type/label
            joint_type = inspection_data.get('label', 'Unknown Inspection')
            
            # Insert into database
            self.cursor.execute('''
                INSERT INTO inspection_records 
                (timestamp, joint_type, confidence, temperature, distance, 
                 line_number, pole_number, ambient_temperature, weather_conditions, inspector_name)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                current_timestamp,
                joint_type,
                confidence,
                temperature_str,
                distance,
                line_number,
                pole_number,
                ambient_temperature,
                weather_conditions,
                inspector_name
            ))
            
            self.conn.commit()
            logging.info(f"✅ Inspection saved: {joint_type} (Confidence: {confidence:.1%})")
            return True
            
        except sqlite3.Error as e:
            logging.error(f"❌ Database save error: {e}")
            return False
        except Exception as e:
            logging.error(f"❌ Unexpected error during save: {e}")
            return False
    
    def save_detection(self, detection_info, form_data=None):
        """
        Save automatic detection information to database
        
        Args:
            detection_info (dict): Detection information from AI model
            form_data (dict, optional): Additional form data
        """
        return self.save_inspection(detection_info, form_data)
    
    def save_manual_inspection(self, detection_info, form_data=None):
        """
        Save manual inspection record to database
        
        Args:
            detection_info (dict): Manual inspection data
            form_data (dict, optional): Form data from inspection details
        """
        return self.save_inspection(detection_info, form_data)
    
    def get_recent_records(self, limit=100):
        """
        Get recent inspection records
        
        Args:
            limit (int): Number of records to retrieve
            
        Returns:
            list: List of inspection records
        """
        try:
            self.cursor.execute('''
                SELECT timestamp, joint_type, confidence, temperature, 
                       line_number, pole_number, inspector_name
                FROM inspection_records 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            records = self.cursor.fetchall()
            logging.info(f"📊 Retrieved {len(records)} recent records")
            return records
            
        except sqlite3.Error as e:
            logging.error(f"❌ Database read error: {e}")
            return []
    
    def get_all_records(self):
        """
        Get all inspection records
        
        Returns:
            list: List of all inspection records
        """
        try:
            self.cursor.execute('''
                SELECT timestamp, joint_type, confidence, temperature, 
                       distance, line_number, pole_number, ambient_temperature, 
                       weather_conditions, inspector_name
                FROM inspection_records
                ORDER BY timestamp DESC
            ''')
            
            records = self.cursor.fetchall()
            logging.info(f"📊 Retrieved {len(records)} total records")
            return records
            
        except sqlite3.Error as e:
            logging.error(f"❌ Database read error: {e}")
            return []
    
    def get_records_by_date_range(self, start_date, end_date):
        """
        Get records within a specific date range
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            list: List of records within the date range
        """
        try:
            self.cursor.execute('''
                SELECT timestamp, joint_type, confidence, temperature, 
                       line_number, pole_number, inspector_name
                FROM inspection_records 
                WHERE date(timestamp) BETWEEN ? AND ?
                ORDER BY timestamp DESC
            ''', (start_date, end_date))
            
            records = self.cursor.fetchall()
            logging.info(f"📊 Retrieved {len(records)} records from {start_date} to {end_date}")
            return records
            
        except sqlite3.Error as e:
            logging.error(f"❌ Database date range query error: {e}")
            return []
    
    def get_statistics(self):
        """
        Get database statistics
        
        Returns:
            dict: Statistics about the database
        """
        try:
            # Total records
            self.cursor.execute("SELECT COUNT(*) FROM inspection_records")
            total_records = self.cursor.fetchone()[0]
            
            # Records by joint type
            self.cursor.execute('''
                SELECT joint_type, COUNT(*) 
                FROM inspection_records 
                GROUP BY joint_type 
                ORDER BY COUNT(*) DESC
            ''')
            records_by_type = dict(self.cursor.fetchall())
            
            # Latest record timestamp
            self.cursor.execute("SELECT MAX(timestamp) FROM inspection_records")
            latest_record = self.cursor.fetchone()[0]
            
            return {
                'total_records': total_records,
                'records_by_type': records_by_type,
                'latest_record': latest_record
            }
            
        except sqlite3.Error as e:
            logging.error(f"❌ Database statistics error: {e}")
            return {}
    
    def clear_records(self):
        """
        Clear all inspection records from database
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.cursor.execute("DELETE FROM inspection_records")
            self.conn.commit()
            logging.info("🗑️ All inspection records cleared from database")
            return True
            
        except sqlite3.Error as e:
            logging.error(f"❌ Database clear error: {e}")
            return False
    
    def export_to_csv(self, filename):
        """
        Export all records to CSV file
        
        Args:
            filename (str): Output CSV filename
            
        Returns:
            tuple: (success, message)
        """
        try:
            records = self.get_all_records()
            if not records:
                return False, "No records to export"
            
            with open(filename, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                # Write header
                writer.writerow([
                    "Timestamp", "JointType", "Confidence", "Temperature", "Distance", 
                    "LineNumber", "PoleNumber", "AmbientTemperature", "WeatherConditions", "InspectorName"
                ])
                
                # Write records
                for record in records:
                    writer.writerow(record)
            
            success_message = f"✅ Exported {len(records)} records to {filename}"
            logging.info(success_message)
            return True, success_message
            
        except Exception as e:
            error_message = f"❌ Export error: {e}"
            logging.error(error_message)
            return False, error_message
    
    def backup_database(self, backup_path):
        """
        Create a backup of the database
        
        Args:
            backup_path (str): Path for the backup file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            import shutil
            shutil.copy2(self.config.DATABASE_PATH, backup_path)
            logging.info(f"💾 Database backed up to: {backup_path}")
            return True
        except Exception as e:
            logging.error(f"❌ Database backup error: {e}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logging.info("🔴 Database connection closed")

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()