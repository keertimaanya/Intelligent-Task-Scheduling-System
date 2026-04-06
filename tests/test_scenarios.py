"""
test_scenarios.py — Run all 3 difficulty levels with a simple greedy agent.

The "agent" here is just a function that picks the highest-priority
pending task and assigns it to the first compatible idle machine.
This shows how the same strategy performs differently at each level.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.scheduler_env import SchedulerEnv
from env.scenarios import generate_easy, generate_medium, generate_hard
from env.models import PENDING


def greedy_agent(obs, env):
    """
    Simple greedy strategy:
    1. Look at all pending tasks, sorted by priority (highest first)
    2. For each, find a compatible idle machine
    3. If found, assign it. If not, WAIT.

    This is NOT an RL agent - it's a baseline to test the environment.
    """
    # Collect pending tasks with their indices
    pending = []
    for i, task in enumerate(obs["tasks"]):
        if task["status"] == PENDING and task["task_id"] != 0:
            pending.append((i, task))

    # Sort by priority (highest first), then by deadline (earliest first)
    pending.sort(key=lambda x: (-x[1]["priority"], x[1]["deadline"]))

    # Try to assign the highest-priority pending task
    for task_idx, task in pending:
        for m_idx, machine in enumerate(obs["machines"]):
            if (not machine["is_busy"] and
                    machine["machine_type"] == task["required_machine_type"]):
                # Found a match! Encode the action
                action = task_idx * len(obs["machines"]) + m_idx
                return action

    # Nothing to assign - WAIT
    return env.WAIT_ACTION


def run_episode(level_name, task_configs, machine_configs):
    """Run one full episode and print the results."""
    env = SchedulerEnv(task_configs, machine_configs, max_time=50, max_steps=200)
    obs = env.reset()
    done = False
    step = 0

    print(f"\n{'='*60}")
    print(f"  {level_name}")
    print(f"  Tasks: {len(task_configs)} | Machines: {len(machine_configs)}")
    print(f"{'='*60}")

    while not done:
        action = greedy_agent(obs, env)
        obs, reward, done, info = env.step(action)
        step += 1

        # Only print steps where something interesting happened
        if reward != 0 or done:
            events = info.get("events", [])
            print(f"  Step {step:2d} | Time {obs['current_time']:2d} | Reward: {reward:+.1f}")
            for e in events:
                print(f"           {e}")

    # Final summary
    print(f"\n  {'-'*50}")
    total = info.get("total_reward", 0)
    tasks_done = obs["num_completed_tasks"]
    tasks_missed = obs["num_missed_tasks"]
    print(f"  TOTAL REWARD:  {total:+.0f}")
    print(f"  Completed:     {tasks_done}")
    print(f"  Missed:        {tasks_missed}")
    print(f"  Steps taken:   {step}")

    if info.get("terminated"):
        print(f"  Ended:         Naturally (all tasks resolved)")
    elif info.get("truncated"):
        print(f"  Ended:         Truncated (time/step limit)")

    return total


def main():
    print("\n" + "#" * 60)
    print("  SCHEDULER ENVIRONMENT - 3 DIFFICULTY LEVELS")
    print("  Agent: Greedy (highest priority first)")
    print("#" * 60)

    # Run all three levels
    scores = {}
    for name, generator in [
        ("EASY", generate_easy),
        ("MEDIUM", generate_medium),
        ("HARD", generate_hard),
    ]:
        tasks, machines = generator()
        scores[name] = run_episode(f"Level: {name}", tasks, machines)

    # Comparison
    print(f"\n{'='*60}")
    print(f"  COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Level':<10} {'Reward':>8}")
    print(f"  {'-'*20}")
    for name, score in scores.items():
        bar = "#" * max(0, int(score / 5))
        print(f"  {name:<10} {score:>+8.0f}  {bar}")

    print()


if __name__ == "__main__":
    main()
