import os
from flask import Flask
from growthlens.config import Config
from growthlens.models import db
from services.data_manager import data_manager

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)

    # Ensure the instance folder exists for SQLite
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    # Initialize extensions
    db.init_app(app)

    # Create SQLite database tables if they do not exist
    with app.app_context():
        db.create_all()

    # Initialize the DataManager (loads and merges CSVs in memory)
    data_manager.init_app(app)

    # Import blueprints inside factory to avoid circular imports
    from growthlens.blueprints.landing import landing_bp
    from growthlens.blueprints.upload import upload_bp
    from growthlens.blueprints.dashboard import dashboard_bp
    from growthlens.blueprints.revenue import revenue_bp
    from growthlens.blueprints.customers import customers_bp
    from growthlens.blueprints.retention import retention_bp
    from growthlens.blueprints.reviews import reviews_bp
    from growthlens.blueprints.action_center import action_center_bp
    from growthlens.blueprints.forecasting import forecasting_bp
    from growthlens.blueprints.data_overview import data_overview_bp

    # Register blueprints at root, using distinct prefixes in routes
    app.register_blueprint(landing_bp)
    app.register_blueprint(upload_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(revenue_bp)
    app.register_blueprint(customers_bp)
    app.register_blueprint(retention_bp)
    app.register_blueprint(reviews_bp)
    app.register_blueprint(action_center_bp)
    app.register_blueprint(forecasting_bp)
    app.register_blueprint(data_overview_bp)

    # Context processor to dynamically fetch Dataset Status across templates using DataManager
    @app.context_processor
    def inject_dataset_status():
        def get_dataset_status():
            diagnostics = data_manager.get_diagnostics()
            health = data_manager.get_dataset_summary()
            
            # Count loaded files vs expected
            loaded_count = sum(1 for name, summary in health.items() if summary.get('status') == 'Loaded')
            total_count = len(health) if health else 9
            
            files_found_str = f"{loaded_count} / {total_count}"
            is_ready = diagnostics.get("load_success", False)
            
            if is_ready:
                ready_status = "Ready"
            elif loaded_count > 0:
                ready_status = "Partial Data"
            else:
                ready_status = "Missing Dataset"
                
            # Try to get latest modified time
            data_dir = app.config.get('DATA_FOLDER')
            last_refresh = "N/A"
            if data_dir and os.path.exists(data_dir) and health:
                mod_times = []
                for name, summary in health.items():
                    path = os.path.join(data_dir, summary['file_name'])
                    if os.path.exists(path):
                        try:
                            mod_times.append(os.path.getmtime(path))
                        except OSError:
                            pass
                if mod_times:
                    from datetime import datetime
                    last_refresh = datetime.fromtimestamp(max(mod_times)).strftime('%Y-%m-%d %H:%M:%S')
                    
            return {
                'detected': os.path.exists(data_dir) if data_dir else False,
                'files_found': files_found_str,
                'last_refresh': last_refresh,
                'ready_status': ready_status,
                'is_ready': is_ready
            }
        
        return dict(dataset_status=get_dataset_status())

    return app
