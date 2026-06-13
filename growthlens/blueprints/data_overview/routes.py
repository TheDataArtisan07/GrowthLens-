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
