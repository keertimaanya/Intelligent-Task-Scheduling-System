"""
baseline_llm.py - OpenAI API Inference Baseline

This script runs an OpenAI model (e.g. gpt-4o-mini) against the environment.
It reads API credentials from the OPENAI_API_KEY environment variable.
It runs through all 3 task difficulties and produces reproducible baseline scores.

Usage:
    set OPENAI_API_KEY=sk-xxxx
    python agents/baseline_llm.py
"""

import sys
import os
import json
import time

# Add root to python path to import our environment & graders
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI
from pydantic import BaseModel
from env.scheduler_env import SchedulerEnv
from env.scenarios import generate_easy, generate_medium, generate_hard
from graders.grader import Grader

# Initialize OpenAI Client (automatically uses OPENAI_API_KEY environment var)
# Provide a friendly exit if the key is missing.
try:
    client = OpenAI()
    # verify key existence
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Testing missing key")
except Exception:
    print("❌ ERROR: OPENAI_API_KEY environment variable is missing or invalid.")
    print("   Please set it using: set OPENAI_API_KEY=your-key-here")
    sys.exit(1)

# You can change the model to 'gpt-4o' for smarter reasoning
MODEL_NAME = "gpt-4o-mini"

def run_level(level: str) -> dict:
    print(f"\n{'='*60}")
    print(f" LLM INFERENCE: Level {level.upper()}")
    print(f"{'='*60}")

    # Generate scenario
    if level == "easy":
        tasks, machines = generate_easy()
    elif level == "medium":
        tasks, machines = generate_medium()
    elif level == "hard":
        tasks, machines = generate_hard()
    else:
        raise ValueError("Invalid level")

    env = SchedulerEnv(tasks, machines)
    obs = env.reset()
    
    done = False
    step_count = 0
    invalid_actions = 0
    total_reward = 0.0

    print("Playing episode...")

    while not done:
        step_count += 1
        
        # Build prompt payload
        system_instruction = f"""
You are an expert AI Scheduler Agent playing a Reinforcement Learning Environment.
Your goal is to maximize reward by assigning Tasks to Machines or Waiting.

Current Observation State:
{json.dumps(obs, indent=2)}

Rule 1: Time only advances when you select the WAIT_ACTION (action ID: {env.WAIT_ACTION}).
Rule 2: If machines are busy or you want to pass time, select the WAIT_ACTION.
Rule 3: To assign a task, output its mapped action ID: (task_index * number_of_machines + machine_index).
        There are {len(machines)} machines. Task ID 1 is at index 0, Task ID 2 is at index 1, etc.
        Example: Task 1 to Machine 1 -> Action 0. Task 1 to Machine 2 -> Action 1.

You must reply with ONLY a raw JSON object containing the integer key 'action'.
Example expected output:
{{"action": {env.WAIT_ACTION}}}
"""

        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                response_format={ "type": "json_object" },
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": "Analyze the state and provide your JSON 'action'."}
                ],
                temperature=0.0 # Deterministic decisions
            )
            
            content = response.choices[0].message.content
            action_data = json.loads(content)
            action = int(action_data.get("action", env.WAIT_ACTION))
            
        except Exception as e:
            print(f" LLM parsing error: {e}. Defaulting to WAIT.")
            action = env.WAIT_ACTION

        # Take specific action
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Track error logic for the grader
        if "error" in info:
            invalid_actions += 1

        # Print trace so user can watch it
        wait_text = "WAIT" if action == env.WAIT_ACTION else f"ASSIGN Action={action}"
        err_text = "" if "error" not in info else f" (ERR: {info['error']})"
        evt_text = "" if "events" not in info else f" | Evts: {info['events']}"
        print(f"  Step {step_count:02d} | LLM chose: {wait_text}{err_text}{evt_text}")

        # Protect against infinite LLM loops making invalid actions forever
        if step_count > max(50, len(tasks) * 5):
            print("  ⚠️ LLM hit maximum step limit (looping invalid actions). Ending episode.")
            break
            
        # Optional: slight delay to avoid hitting rate limits instantly
        time.sleep(0.1)

    print(f"\n-- Episode complete! Raw Reward: {total_reward} --")

    # Evaluate using the Grader component
    grader = Grader(level)
    score_report = grader.evaluate(env, invalid_actions)
    
    return score_report

def main():
    levels = ["easy", "medium", "hard"]
    final_scores = {}

    for lvl in levels:
        score_report = run_level(lvl)
        final_scores[lvl.upper()] = score_report

    print("\n\n" + "="*60)
    print("  FINAL LLM INFERENCE BASELINE SCORES")
    print("="*60)
    for lvl, report in final_scores.items():
        print(f"\nLevel: {lvl}")
        print(f"  WCR (Completion Rate): {report['WCR']}")
        print(f"  PWE (Earliness):       {report['PWE']}")
        print(f"  ER  (Efficiency):      {report['ER']}")
        print(f"  AA  (Action Accuracy): {report['AA']}")
        print(f"  OVERALL SCORE:         {report['Overall']}")

    print("\nBenchmark Evaluation Complete.")

if __name__ == "__main__":
    main()
