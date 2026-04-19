import json
import os
import time

LOG_FILE = "pipeline_metrics.json"

def log_stage(stage_name, duration, output_dim):
    data = {}
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            try:
                data = json.load(f)
            except:
                data = {}
    
    data[stage_name] = {
        "Processing Time": f"{duration:.1f} sec" if duration > 1 else f"{duration*1000:.1f} ms",
        "Output Dimensions": output_dim
    }
    
    with open(LOG_FILE, "w") as f:
        json.dump(data, f, indent=4)

class StageTimer:
    def __init__(self, stage_name):
        self.stage_name = stage_name
        self.start = None
    
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        # We don't strictly need duration here if we log inside the block
        pass
    
    def log(self, dim):
        duration = time.time() - self.start
        log_stage(self.stage_name, duration, dim)
