import os
import datetime

def parse_hopson_one(basename):
    ts_start_str = basename.split('_')[0]
    ts_start = datetime.datetime.strptime(ts_start_str, '%Y-%m-%d-%H%M%S%f')
    byte_per_sec = 2147484644 / 8492

    if os.path.exists(os.path.join('../data/input', basename)):
        file_size = os.path.getsize(os.path.join('../data/input', basename))
        duration = int(file_size / byte_per_sec) 
    else:
        file_size = None
        duration = None
        
    camera_id = basename.split('_')[1].split('-')[1]
    floor_id = basename.split('_')[1].split('-')[0]
    
    return ts_start, file_size, byte_per_sec, duration, camera_id, floor_id
    
