"""
test_env.py — Run the exact worked example from the design doc.

This replays the 4-task, 2-machine scenario step by step,
printing the observation, reward, and events at each step.

Expected total reward: +110
"""

import sys
import os

# Add project root to path so we can import env
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env.scheduler_env import SchedulerEnv


def main():
    # ─────────────────────────────────────────
    #  Setup: Exact same scenario from design doc
    # ─────────────────────────────────────────
    task_configs = [
        # T1: high priority, arrives at time 0, needs Type A machine
        {"task_id": 1, "duration": 2, "deadline": 5, "priority": 3,
         "required_machine_type": 1, "arrival_time": 0},
        # T2: medium priority, arrives at time 0, needs Type B machine
        {"task_id": 2, "duration": 1, "deadline": 3, "priority": 2,
         "required_machine_type": 2, "arrival_time": 0},
        # T3: low priority, arrives at time 3 (SURPRISE!), needs Type A
        {"task_id": 3, "duration": 3, "deadline": 7, "priority": 1,
         "required_machine_type": 1, "arrival_time": 3},
        # T4: high priority, arrives at time 3 (SURPRISE!), needs Type B
        {"task_id": 4, "duration": 2, "deadline": 6, "priority": 3,
         "required_machine_type": 2, "arrival_time": 3},
    ]

    machine_configs = [
        {"machine_id": 1, "machine_type": 1},   # M1: Type A
        {"machine_id": 2, "machine_type": 2},   # M2: Type B
    ]

    env = SchedulerEnv(task_configs, machine_configs, max_time=50, max_steps=200)

    # ─────────────────────────────────────────
    #  Reset — Start the episode
    # ─────────────────────────────────────────
    obs = env.reset()
    print("=" * 60)
    print("  INTELLIGENT TASK SCHEDULING — TEST RUN")
    print("=" * 60)
    print(f"\nInitial observation:")
    print(f"  Time: {obs['current_time']}")
    print(f"  Pending: {obs['num_pending_tasks']}, Active: {obs['num_active_tasks']}")
    print(f"  Visible tasks: {len([t for t in obs['tasks'] if t['task_id'] != 0])}")
    env.debug_state()

    # ─────────────────────────────────────────
    #  Step 1: Assign T1 → M1 (action = task_0 * 2 + machine_0 = 0)
    # ─────────────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 1: Assign T1 → M1")
    obs, reward, done, info = env.step(0)    # task_index=0, machine_index=0
    print(f"  Reward: {reward}, Done: {done}")
    print(f"  Info: {info}")

    # ─────────────────────────────────────────
    #  Step 2: Assign T2 → M2 (action = task_1 * 2 + machine_1 = 3)
    # ─────────────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 2: Assign T2 → M2")
    obs, reward, done, info = env.step(3)    # task_index=1, machine_index=1
    print(f"  Reward: {reward}, Done: {done}")
    print(f"  Info: {info}")

    # ─────────────────────────────────────────
    #  Step 3: WAIT (both machines busy)
    # ─────────────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 3: WAIT")
    obs, reward, done, info = env.step(env.WAIT_ACTION)
    print(f"  Reward: {reward}, Done: {done}")
    print(f"  Info: {info}")
    print(f"  Time is now: {obs['current_time']}")

    # ─────────────────────────────────────────
    #  Step 4: WAIT (T2 done, nothing to assign on M2)
    # ─────────────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 4: WAIT")
    obs, reward, done, info = env.step(env.WAIT_ACTION)
    print(f"  Reward: {reward}, Done: {done}")
    print(f"  Info: {info}")
    print(f"  Time is now: {obs['current_time']}")

    # ─────────────────────────────────────────
    #  Step 5: WAIT (waiting for T3 and T4 to arrive)
    # ─────────────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 5: WAIT (waiting for arrivals)")
    obs, reward, done, info = env.step(env.WAIT_ACTION)
    print(f"  Reward: {reward}, Done: {done}")
    print(f"  Info: {info}")
    print(f"  Time is now: {obs['current_time']}")
    print(f"  Pending tasks: {obs['num_pending_tasks']}  ← T3 & T4 just appeared!")

    # ─────────────────────────────────────────
    #  Step 6: Assign T4 → M2 (task_index=3, machine_index=1 → 3*2+1=7)
    # ─────────────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 6: Assign T4 → M2")
    obs, reward, done, info = env.step(7)    # task_index=3, machine_index=1
    print(f"  Reward: {reward}, Done: {done}")
    print(f"  Info: {info}")

    # ─────────────────────────────────────────
    #  Step 7: Assign T3 → M1 (task_index=2, machine_index=0 → 2*2+0=4)
    # ─────────────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 7: Assign T3 → M1")
    obs, reward, done, info = env.step(4)    # task_index=2, machine_index=0
    print(f"  Reward: {reward}, Done: {done}")
    print(f"  Info: {info}")

    # ─────────────────────────────────────────
    #  Step 8: WAIT (T4 finishes first)
    # ─────────────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 8: WAIT")
    obs, reward, done, info = env.step(env.WAIT_ACTION)
    print(f"  Reward: {reward}, Done: {done}")
    print(f"  Info: {info}")
    print(f"  Time is now: {obs['current_time']}")

    # ─────────────────────────────────────────
    #  Step 9: WAIT (T3 finishes, episode ends)
    # ─────────────────────────────────────────
    print("\n" + "─" * 60)
    print("STEP 9: WAIT")
    obs, reward, done, info = env.step(env.WAIT_ACTION)
    print(f"  Reward: {reward}, Done: {done}")
    print(f"  Info: {info}")

    # ─────────────────────────────────────────
    #  Final state
    # ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  EPISODE COMPLETE")
    print("=" * 60)
    env.debug_state()
    print(f"\n  TOTAL REWARD: {info.get('total_reward', '?')}")
    print(f"  EXPECTED:     110")

    # ─────────────────────────────────────────
    #  Test invalid actions too
    # ─────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  BONUS: TESTING INVALID ACTIONS")
    print("=" * 60)

    # Reset for a clean test
    env.reset()

    # Try assigning T1 to M2 (wrong machine type: T1 needs type 1, M2 is type 2)
    print("\nTest: Assign T1 → M2 (wrong type)")
    obs, reward, done, info = env.step(1)    # task_index=0, machine_index=1
    print(f"  Reward: {reward} (expected: -2)")
    print(f"  Error: {info.get('error', 'none')}")

    # Assign T1 → M1 (valid)
    env.step(0)

    # Try assigning T1 again (already running)
    print("\nTest: Assign T1 again (already running)")
    obs, reward, done, info = env.step(0)    # Same action
    print(f"  Reward: {reward} (expected: -2)")
    print(f"  Error: {info.get('error', 'none')}")

    print("\n✅ All tests completed!")


if __name__ == "__main__":
    main()
