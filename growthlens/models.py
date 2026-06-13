from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class MetadataLog(db.Model):
    """
    Lightweight model for tracking dataset status, refresh events, and execution logs.
    """
    __tablename__ = 'metadata_logs'
    
    id = db.Column(db.Integer, primary_key=True)
    event_type = db.Column(db.String(50), nullable=False)  # 'DATASET_REFRESH', 'API_ACCESS', etc.
    description = db.Column(db.String(255), nullable=True)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<MetadataLog {self.event_type} at {self.timestamp}>"

class AppSetting(db.Model):
    """
    Key-value storage for application level configuration (e.g., custom UI features, threshold limits).
    """
    __tablename__ = 'app_settings'
    
    key = db.Column(db.String(50), primary_key=True)
    value = db.Column(db.String(255), nullable=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<AppSetting {self.key}={self.value}>"
