import tkinter as tk
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """Main application entry point"""
    try:
        # Create and run the application
        root = tk.Tk()
        
        # Import here to avoid circular imports
        from gui import PowerLineInspectorGUI
        
        app = PowerLineInspectorGUI(root)
        
        # Center the window on screen
        root.update_idletasks()
        width = root.winfo_width()
        height = root.winfo_height()
        x = (root.winfo_screenwidth() // 2) - (width // 2)
        y = (root.winfo_screenheight() // 2) - (height // 2)
        root.geometry(f"+{x}+{y}")
        
        # Handle window close event
        root.protocol("WM_DELETE_WINDOW", app.on_closing)
        
        logging.info("🚀 Power Line Inspection System Started")
        root.mainloop()
        
    except Exception as e:
        logging.error(f"❌ Application failed to start: {e}")
        raise

if __name__ == "__main__":
    main()