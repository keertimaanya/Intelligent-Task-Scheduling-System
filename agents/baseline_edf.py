"""
baseline_edf.py - Earliest Deadline First (EDF) baseline agent.

Strategy:
  Among all PENDING tasks that have a compatible idle machine,
  pick the one with the EARLIEST DEADLINE and assign it.

  If no assignment is possible, WAIT.

Why EDF works:
  - Tasks closest to their deadline are most urgent
  - By always doing the most urgent task first, we minimize
    the chance of missing any deadline
  - It's optimal for single-machine scheduling (proven in theory)

Where EDF fails:
  - It ignores PRIORITY. A low-priority task due in 1 hour gets
    scheduled before a high-priority task due in 2 hours.
    The high-priority task was worth 3x more reward!
  - It can't handle TRADE-OFFS. When not all tasks can be completed,
    EDF will try to save every task equally instead of sacrificing
    low-value ones to protect high-value ones.
  - It's GREEDY (no lookahead). It never reserves a machine for a
    more important task arriving later.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.scheduler_env import SchedulerEnv
from env.scenarios import generate_easy, generate_medium, generate_hard
from env.models import PENDING


def edf_agent(obs, env):
    """
    Earliest Deadline First agent.

    1. Look at all pending tasks (visible, status == PENDING)
    2. Sort by deadline (earliest first)
    3. For each task, check if a compatible idle machine exists
    4. Assign the first valid match
    5. If nothing can be assigned, WAIT

    This is purely reactive - no planning, no priority awareness.
    """
    # Collect pending tasks with their slot index
    pending = []
    for i, task in enumerate(obs["tasks"]):
        if task["status"] == PENDING and task["task_id"] != 0:
            pending.append((i, task))

    # Sort by deadline: earliest deadline first (the core of EDF)
    pending.sort(key=lambda x: x[1]["deadline"])

    # Try to assign the most urgent task to a compatible machine
    for task_idx, task in pending:
        for m_idx, machine in enumerate(obs["machines"]):
            if (not machine["is_busy"] and
                    machine["machine_type"] == task["required_machine_type"]):
                # Encode action: task_index * num_machines + machine_index
                action = task_idx * len(obs["machines"]) + m_idx
                return action, task, machine

    # Nothing assignable - WAIT
    return env.WAIT_ACTION, None, None


def run_episode_verbose(level_name, task_configs, machine_configs):
    """Run one episode with step-by-step output."""
    env = SchedulerEnv(task_configs, machine_configs, max_time=50, max_steps=200)
    obs = env.reset()
    done = False
    step_num = 0

    print(f"\n{'='*60}")
    print(f"  {level_name}")
    print(f"  Tasks: {len(task_configs)} | Machines: {len(machine_configs)}")
    print(f"  Strategy: Earliest Deadline First (EDF)")
    print(f"{'='*60}")

    # Show initial state
    visible = [t for t in obs["tasks"] if t["task_id"] != 0]
    print(f"\n  Time 0 - Initial state:")
    print(f"  Visible tasks: {len(visible)}")
    for t in visible:
        print(f"    T{t['task_id']}: dur={t['duration']}, "
              f"deadline={t['deadline']}, pri={t['priority']}, "
              f"type={t['required_machine_type']}")

    while not done:
        action, chosen_task, chosen_machine = edf_agent(obs, env)
        obs, reward, done, info = env.step(action)
        step_num += 1

        # Describe what happened
        if action == env.WAIT_ACTION:
            events = info.get("events", [])
            print(f"\n  Step {step_num:2d} | WAIT")
            print(f"    Time -> {obs['current_time']}")
            for e in events:
                print(f"    >> {e}")
            if reward != 0:
                print(f"    Reward: {reward:+.0f}")
        else:
            # Assignment action
            if chosen_task and chosen_machine:
                print(f"\n  Step {step_num:2d} | ASSIGN "
                      f"T{chosen_task['task_id']} -> "
                      f"M{chosen_machine['machine_id']}")
                print(f"    Why: T{chosen_task['task_id']} has "
                      f"earliest deadline ({chosen_task['deadline']})")
            if reward != 0:
                events = info.get("events", [])
                for e in events:
                    print(f"    >> {e}")
                print(f"    Reward: {reward:+.0f}")

    # Final summary
    total = info.get("total_reward", 0)
    print(f"\n  {'-'*50}")
    print(f"  RESULT: Reward = {total:+.0f}")
    print(f"    Completed: {obs['num_completed_tasks']}")
    print(f"    Missed:    {obs['num_missed_tasks']}")
    print(f"    Steps:     {step_num}")

    return total


def main():
    print("\n" + "#" * 60)
    print("  BASELINE AGENT: Earliest Deadline First (EDF)")
    print("#" * 60)

    results = {}
    for name, gen in [("EASY", generate_easy),
                      ("MEDIUM", generate_medium),
                      ("HARD", generate_hard)]:
        tasks, machines = gen()
        results[name] = run_episode_verbose(
            f"Level: {name}", tasks, machines
        )

    # Summary comparison
    print(f"\n{'='*60}")
    print(f"  EDF RESULTS ACROSS ALL LEVELS")
    print(f"{'='*60}")
    for name, score in results.items():
        print(f"  {name:8s}  {score:+6.0f}")
    print()


if __name__ == "__main__":
    main()
