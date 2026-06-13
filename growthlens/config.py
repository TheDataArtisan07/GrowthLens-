import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'growthlens-dev-secret-key-198273')
    
    # Base directory of the project
    BASE_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    
    # Database configuration
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        'DATABASE_URL', 
        f"sqlite:///{os.path.join(BASE_DIR, 'instance', 'growthlens.db')}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Data directory containing Olist dataset CSV files
    DATA_FOLDER = os.path.join(BASE_DIR, 'data')
