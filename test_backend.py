import urllib.request
import json
import uvicorn
import threading
import time
from app import app

def run_server():
    uvicorn.run(app, host="127.0.0.1", port=8001, log_level="error")

def test_inference():
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    
    time.sleep(5)  # give server time to start
    
    payload = {
        "TransactionAmt": 150.00,
        "TransactionDT": 80000,
        "V2_te": 0.5
    }
    
    data = json.dumps(payload).encode('utf-8')
    req = urllib.request.Request("http://127.0.0.1:8001/predict", data=data, headers={'Content-Type': 'application/json'})
    
    print("Sending payload...")
    try:
        with urllib.request.urlopen(req) as response:
            print("Response status:", response.status)
            body = response.read()
            print("Response JSON:", json.loads(body))
    except urllib.error.HTTPError as e:
        print("HTTP Error:", e.code, e.read().decode())
    except Exception as e:
        print("Error:", e)

if __name__ == "__main__":
    test_inference()
