import json
from datetime import datetime, timezone
from pathlib import Path 

LOG_PATH = Path("logs/app.jsonl")

def log(event: str, trace_id: str = None, **kwargs) -> None:
    record = {
        "timestamp" : datetime.now(timezone.utc).isoformat(),
        "trace_id" : trace_id,
        "event" : event,
        **kwargs
    }

    LOG_PATH.parent.mkdir(exist_ok=True)
    with LOG_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(json.dumps(record))
    
    
