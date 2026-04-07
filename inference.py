"""
inference.py - OpenEnv Baseline Inference Script

Uses the OpenAI API client to run a model against the environment.
Reads credentials from environment variables:
  - API_BASE_URL  : The API endpoint for the LLM
  - MODEL_NAME    : The model identifier (e.g. gpt-4o-mini)
  - HF_TOKEN      : Your Hugging Face / API key

Produces a reproducible baseline score on all 3 tasks.
Emits structured stdout logs in [START], [STEP], [END] format.
"""

import os
import sys
import json
import time
import requests

from openai import OpenAI

# ─────────────────────────────────────────────
#  Configuration from environment variables
# ─────────────────────────────────────────────

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "") or os.environ.get("OPENAI_API_KEY", "")

if not HF_TOKEN:
    print("ERROR: HF_TOKEN (or OPENAI_API_KEY) environment variable is not set.")
    print("  Set it with:  $env:HF_TOKEN='your-key-here'")
    sys.exit(1)

# Initialize OpenAI-compatible client using the mandatory env vars
client = OpenAI(
    api_key=HF_TOKEN,
    base_url=API_BASE_URL,
)

# Environment server URL (the FastAPI server must be running)
ENV_URL = os.environ.get("ENV_URL", "http://localhost:7860")

# ─────────────────────────────────────────────
#  Helper: call environment endpoints
# ─────────────────────────────────────────────

def env_reset(level: str) -> dict:
    r = requests.post(f"{ENV_URL}/reset", json={"level": level})
    r.raise_for_status()
    return r.json()

def env_step(action: int) -> dict:
    r = requests.post(f"{ENV_URL}/step", json={"action": action})
    r.raise_for_status()
    return r.json()

def env_state() -> dict:
    r = requests.get(f"{ENV_URL}/state")
    r.raise_for_status()
    return r.json()

def env_grade(invalid_actions: int = 0) -> dict:
    r = requests.post(f"{ENV_URL}/grade", params={"invalid_actions": invalid_actions})
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────
#  Run one task
# ─────────────────────────────────────────────

def run_task(task_id: str) -> dict:
    # [START] log
    print(f"[START] task={task_id} env=intelligent-task-scheduling model={MODEL_NAME}", flush=True)

    # Reset environment for this task/level
    reset_data = env_reset(task_id)
    obs = reset_data["observation"]
    wait_action = reset_data["wait_action"]

    done = False
    step_num = 0
    invalid_actions = 0
    episode_rewards = []

    try:
        while not done:
            step_num += 1

            # Build LLM prompt
            num_machines = len(obs.get("machines", []))
            system_prompt = f"""You are an expert AI Scheduler Agent.
Your goal: maximize reward by assigning Tasks to Machines before deadlines.

Current state:
{json.dumps(obs, indent=2)}

Rules:
- WAIT action ID = {wait_action}. This advances time.
- To assign: action = task_index * {num_machines} + machine_index
- Task IDs start at 1, indices start at 0. Task 1 = index 0, Task 2 = index 1, etc.
- Only assign PENDING tasks (status=0) to IDLE machines (is_busy=0).
- Match required_machine_type to machine_type.

Reply with ONLY a JSON object: {{"action": <integer>}}"""

            try:
                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": "Choose your next action as JSON."}
                    ],
                    temperature=0.0,
                )
                content = response.choices[0].message.content
                action_data = json.loads(content)
                action = int(action_data.get("action", wait_action))
            except Exception as e:
                action = wait_action

            # Execute step
            step_result = env_step(action)
            obs = step_result["observation"]
            reward = float(step_result["reward"])
            done = step_result["done"]
            info = step_result["info"]
            
            episode_rewards.append(reward)

            if "error" in info:
                invalid_actions += 1
                err_msg = f'"{info["error"]}"'
            else:
                err_msg = "null"

            done_str = "true" if done else "false"
            action_type = "WAIT" if action == wait_action else f"ASSIGN({action})"
            
            # [STEP] log
            print(f"[STEP] step={step_num} action={action_type} reward={reward:.2f} done={done_str} error={err_msg}", flush=True)

            # Safety: prevent infinite loops
            if step_num > 100:
                print(f"[STEP] step={step_num+1} action=FORCE_END reward=0.00 done=true error=\"Max steps exceeded\"", flush=True)
                break

            time.sleep(0.05)

    finally:
        # Grade (must be done after loop ends)
        # Handle case where server errors prevented grading
        try:
             grade_result = env_grade(invalid_actions)
             score = float(grade_result["score"])
        except Exception:
             grade_result = {"score": 0.0, "details": {}}
             score = 0.0
             
        success_str = "true" if score > 0.0 else "false"
        rewards_str = ",".join(f"{r:.2f}" for r in episode_rewards)
        
        # [END] log
        print(f"[END] success={success_str} steps={step_num} score={score:.3f} rewards={rewards_str}", flush=True)

    return grade_result


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    task_ids = ["easy", "medium", "hard"]
    results = {}

    for task_id in task_ids:
        result = run_task(task_id)
        results[task_id] = result

    # Final summary
    print("\n" + "=" * 60)
    print("  INFERENCE RESULTS SUMMARY")
    print("=" * 60)
    for tid, res in results.items():
        print(f"  {tid.upper():8s} | Score: {res['score']:.3f} | Details: {res['details']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
