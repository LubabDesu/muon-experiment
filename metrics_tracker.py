import json
import time
from typing import Dict, List, Optional

class MetricsTracker:
    """Simple metrics accumulator for training experiments."""
    
    def __init__(self, name: str = "experiment"):
        self.name = name
        self.metrics = []
        self.start_time = time.time()
    
    def log(self, step: int, **kwargs):
        """
        Log metrics at a step.
        
        Example: tracker.log(step=100, loss=0.5, cosine_sim=0.85, val_loss=0.55)
        """
        entry = {"step": step, "elapsed_s": time.time() - self.start_time}
        entry.update(kwargs)
        self.metrics.append(entry)
    
    def save(self, filepath: str):
        """Save metrics to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {filepath}")
    
    def get_latest(self, key: str) -> Optional[float]:
        """Get the most recent value for a metric key."""
        if self.metrics:
            return self.metrics[-1].get(key)
        return None
    
    def summary(self) -> Dict:
        """Get summary statistics of recorded metrics."""
        if not self.metrics:
            return {}
        
        summary = {}
        for key in self.metrics[0].keys():
            if key in ["step", "elapsed_s"]:
                continue
            try:
                values = [m.get(key) for m in self.metrics if m.get(key) is not None]
                if values:
                    summary[key] = {
                        "min": min(values),
                        "max": max(values),
                        "mean": sum(values) / len(values),
                        "latest": values[-1]
                    }
            except TypeError:
                pass
        
        return summary
