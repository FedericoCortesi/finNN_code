def format_legend_name(raw_name):
    # Split the string by '_'
    parts = raw_name.split('_')
    
    # Check if format is valid to avoid errors
    if len(parts) >= 5:
        # parts[2] = architecture (e.g., cnn)
        # parts[3] = date range/size (e.g., 100)
        # parts[4] = optimizer (e.g., muon)
        
        # Return formatted string, e.g., "CNN 100 MUON"
        return f"{parts[2]} {parts[3]} {parts[4]}".upper()
    
    return raw_name # Return original if format doesn't match