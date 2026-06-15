from flask import Blueprint, render_template
from services.data_manager import data_manager

data_overview_bp = Blueprint('data_overview', __name__)

@data_overview_bp.route('/data-overview')
def index():
    # Retrieve pre-calculated metrics from memory cache
    diagnostics = data_manager.get_diagnostics()
    health_reports = data_manager.get_dataset_health()
    
    return render_template(
        'data_overview.html', 
        active_page='data_overview',
        diagnostics=diagnostics,
        health_reports=health_reports
    )

@data_overview_bp.route('/requirements')
def requirements():
    import os
    from flask import current_app
    req_path = os.path.join(current_app.root_path, '..', 'requirements.txt')
    req_content = ""
    if os.path.exists(req_path):
        with open(req_path, 'r') as f:
            req_content = f.read()
            
    req_list = [
        {"name": "Flask", "version": ">=3.0.0", "description": "The micro web framework used to build our backend application factory and routing layer."},
        {"name": "Flask-SQLAlchemy", "version": ">=3.1.0", "description": "SQLAlchemy extension for Flask that provides simple database mapping utilities."},
        {"name": "pandas", "version": ">=2.0.0", "description": "High-performance data analysis library used for in-memory merging, RFM parsing, and aggregation calculations."},
        {"name": "plotly", "version": ">=5.15.0", "description": "Interactive, browser-based graphing library used to render all e-commerce dashboards dynamically."},
        {"name": "SQLAlchemy", "version": ">=2.0.0", "description": "Database toolkit and Object-Relational Mapper (ORM) used to model project metadata."}
    ]
    
    return render_template(
        'requirements.html',
        active_page='requirements',
        req_content=req_content,
        req_list=req_list
    )
