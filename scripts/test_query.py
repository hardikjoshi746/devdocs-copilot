import requests
import sys

TOKEN = sys.argv[1] if len(sys.argv) > 1 else ""
question = sys.argv[2] if len(sys.argv) > 2 else "how does job scraping work?"

resp = requests.post(
    "http://localhost:8000/query",
    headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"},
    json={"question": question},
)
import json
print(json.dumps(resp.json(), indent=2))