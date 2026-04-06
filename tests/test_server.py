"""Test the FastAPI server endpoints."""
import requests
import json

BASE = "http://localhost:8000"

def test_reset():
    print("=== POST /reset (easy) ===")
    r = requests.post(f"{BASE}/reset", json={"level": "easy"})
    data = r.json()
    print(f"  Status: {r.status_code}")
    obs = data["observation"]
    print(f"  Time: {obs['current_time']}")
    print(f"  Pending: {obs['num_pending_tasks']}")
    print(f"  WAIT action ID: {data['wait_action']}")
    print()
    return data["wait_action"]

def test_state():
    print("=== GET /state ===")
    r = requests.get(f"{BASE}/state")
    data = r.json()
    print(f"  Status: {r.status_code}")
    print(f"  Done: {data['done']}")
    tasks = data["observation"]["tasks"]
    for t in tasks:
        if t["task_id"] != 0:
            print(f"  T{t['task_id']}: status={t['status']}, dl={t['deadline']}")
    print()

def test_step_assign():
    print("=== POST /step (action=0: T1->M1) ===")
    r = requests.post(f"{BASE}/step", json={"action": 0})
    data = r.json()
    print(f"  Status: {r.status_code}")
    print(f"  Reward: {data['reward']}")
    print(f"  Done: {data['done']}")
    info = data["info"]
    if "assigned" in info:
        print(f"  Assigned: {info['assigned']}")
    if "error" in info:
        print(f"  Error: {info['error']}")
    print()

def test_step_wait(wait_action):
    print(f"=== POST /step (action={wait_action}: WAIT) ===")
    r = requests.post(f"{BASE}/step", json={"action": wait_action})
    data = r.json()
    print(f"  Status: {r.status_code}")
    print(f"  Reward: {data['reward']}")
    print(f"  Time now: {data['observation']['current_time']}")
    for e in data["info"].get("events", []):
        print(f"  Event: {e}")
    print()

def test_error():
    print("=== POST /reset (bad level) ===")
    r = requests.post(f"{BASE}/reset", json={"level": "impossible"})
    print(f"  Status: {r.status_code}")
    print(f"  Error: {r.json()['detail']}")
    print()

# Run all tests
wait_action = test_reset()
test_state()
test_step_assign()
test_step_wait(wait_action)
test_error()
print("All tests passed!")
