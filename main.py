from database import init_db
from gui.app import run_app

if __name__ == "__main__":
    # Initialize Database Schema
    init_db()
    
    # Launch Streamlit GUI App
    run_app()