import tempfile
import os
import zipfile
import pandas as pd
from flask import Blueprint, render_template, request, flash, redirect, url_for, current_app
from services.data_manager import data_manager

upload_bp = Blueprint('upload', __name__)

def map_filename_to_key(filename):
    """
    Maps an uploaded filename to one of the 9 dataset keys.
    Prevents false overlaps (e.g. order_items matching orders).
    """
    fn = filename.lower()
    if 'order_items' in fn or 'items' in fn:
        return 'order_items'
    elif 'payment' in fn:
        return 'payments'
    elif 'customer' in fn:
        return 'customers'
    elif 'review' in fn:
        return 'reviews'
    elif 'product' in fn:
        return 'products'
    elif 'seller' in fn:
        return 'sellers'
    elif 'geo' in fn:
        return 'geolocation'
    elif 'translation' in fn or 'category' in fn:
        return 'category_translation'
    elif 'order' in fn:
        return 'orders'
    return None

@upload_bp.route('/upload', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('upload.html', validated=False)

    # Handle file upload post
    if 'file' not in request.files:
        flash('No file part in the request', 'danger')
        return redirect(request.url)
        
    uploaded_file = request.files['file']
    if uploaded_file.filename == '':
        flash('No file selected', 'danger')
        return redirect(request.url)

    # Ensure uploads directory exists in system temp directory to prevent Flask watchdog reload
    temp_dir = os.path.join(tempfile.gettempdir(), 'growthlens_temp_uploads')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Clean up temp directory first
    for f in os.listdir(temp_dir):
        fp = os.path.join(temp_dir, f)
        if os.path.isfile(fp):
            try:
                os.remove(fp)
            except Exception:
                pass

    filename = uploaded_file.filename
    filepath = os.path.join(temp_dir, filename)
    uploaded_file.save(filepath)

    # 1. Handle ZIP files
    if filename.lower().endswith('.zip'):
        try:
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            # Remove the uploaded ZIP file so it's not scanned as CSV
            os.remove(filepath)
        except Exception as e:
            flash(f'Failed to extract ZIP archive: {str(e)}', 'danger')
            return redirect(request.url)

    # Scan for CSV files in temp directory
    csv_files = []
    for root, dirs, files in os.walk(temp_dir):
        for f in files:
            if f.lower().endswith('.csv'):
                csv_files.append(os.path.join(root, f))

    if not csv_files:
        flash('No CSV files detected in the uploaded upload package.', 'warning')
        return render_template('upload.html', validated=False)

    validation_results = []
    previews = {}
    total_rows = 0
    detected_keys = set()
    
    # Expected schemas from DataManager
    definitions = data_manager.dataset_definitions

    for csv_path in csv_files:
        fname = os.path.basename(csv_path)
        key = map_filename_to_key(fname)
        
        status_info = {
            "file_name": fname,
            "mapped_table": key if key else "Unknown / Unmapped",
            "status": "Invalid",
            "rows": 0,
            "columns": 0,
            "validation_result": "Unmapped table schema"
        }
        
        if key:
            detected_keys.add(key)
            required_cols = definitions[key]["required_columns"]
            try:
                # Read CSV
                df = pd.read_csv(csv_path, nrows=100) # Only load head for validation speed
                rows_count = sum(1 for _ in open(csv_path)) - 1 # Faster row count without full load
                cols_count = len(df.columns)
                
                status_info["rows"] = rows_count
                status_info["columns"] = cols_count
                total_rows += rows_count
                
                # Check columns schema
                missing_cols = [c for c in required_cols if c not in df.columns]
                if missing_cols:
                    status_info["status"] = "Invalid"
                    status_info["validation_result"] = f"Missing required columns: {', '.join(missing_cols)}"
                else:
                    status_info["status"] = "Valid"
                    status_info["validation_result"] = "All required columns verified."
                    
                    # Generate preview dictionary
                    preview_df = df.head(5).fillna('--')
                    previews[fname] = {
                        "headers": list(preview_df.columns),
                        "rows": preview_df.to_dict(orient='records')
                    }
                    
            except Exception as e:
                status_info["status"] = "Invalid"
                status_info["validation_result"] = f"Failed parsing file: {str(e)}"
        
        validation_results.append(status_info)

    # Core table completeness check
    core_keys = {'orders', 'customers', 'order_items', 'payments'}
    valid_detected_keys = {r["mapped_table"] for r in validation_results if r["status"] == "Valid"}
    is_ready = core_keys.issubset(valid_detected_keys)

    # Summary Info
    summary = {
        "files_count": len(validation_results),
        "total_rows": total_rows,
        "detected_tables": len(valid_detected_keys),
        "is_ready": is_ready
    }

    return render_template(
        'upload.html',
        validated=True,
        validation_results=validation_results,
        previews=previews,
        summary=summary
    )
