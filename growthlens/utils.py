import base64
import json
import numpy as np

def plotly_to_json(fig):
    """
    Serializes a Plotly figure to a JSON string, ensuring that all numpy types
    are converted to standard Python types, and decoding any base64-encoded
    binary buffers ('bdata') to avoid compatibility issues with the browser's CDN Plotly.js.
    """
    fig_dict = fig.to_dict()
    
    def decode_node(obj):
        if isinstance(obj, dict):
            # Check for base64 encoded numpy arrays
            if 'dtype' in obj and 'bdata' in obj:
                dtype = obj['dtype']
                bdata = obj['bdata']
                byte_data = base64.b64decode(bdata)
                arr = np.frombuffer(byte_data, dtype=dtype)
                
                # Reshape array if shape is provided
                if 'shape' in obj:
                    shape = tuple(int(x) for x in obj['shape'].split(','))
                    arr = arr.reshape(shape)
                
                # Convert array to list and sanitize NaNs/Infs
                arr_list = arr.tolist()
                def sanitize_list(lst):
                    if isinstance(lst, list):
                        return [sanitize_list(item) for item in lst]
                    elif isinstance(lst, float) and (np.isnan(lst) or np.isinf(lst)):
                        return None
                    return lst
                return sanitize_list(arr_list)
                
            return {k: decode_node(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [decode_node(x) for x in obj]
        elif isinstance(obj, np.ndarray):
            # Convert raw numpy arrays to lists and sanitize
            arr_list = obj.tolist()
            def sanitize_list(lst):
                if isinstance(lst, list):
                    return [sanitize_list(item) for item in lst]
                elif isinstance(lst, float) and (np.isnan(lst) or np.isinf(lst)):
                    return None
                return lst
            return sanitize_list(arr_list)
        elif isinstance(obj, (np.integer, np.floating)):
            val = obj.item()
            if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                return None
            return val
        elif isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        else:
            return obj
            
    cleaned_dict = decode_node(fig_dict)
    return json.dumps(cleaned_dict)
