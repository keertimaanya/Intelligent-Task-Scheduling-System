"""Minimal step trace for EDF agent - avoids terminal width issues."""
import sys; sys.path.insert(0, '.')
from env import *
from env.models import PENDING

def edf(obs, env):
    p = [(i, t) for i, t in enumerate(obs["tasks"])
         if t["status"] == PENDING and t["task_id"] != 0]
    p.sort(key=lambda x: x[1]["deadline"])
    for ti, t in p:
        for mi, m in enumerate(obs["machines"]):
            if not m["is_busy"] and m["machine_type"] == t["required_machine_type"]:
                return ti * len(obs["machines"]) + mi, t["task_id"], m["machine_id"], t["deadline"]
    return env.WAIT_ACTION, 0, 0, 0

def run(name, gen):
    t, m = gen()
    env = SchedulerEnv(t, m)
    obs = env.reset()
    done = False
    step = 0
    print(f"--- {name} ---")
    while not done:
        act, tid, mid, dl = edf(obs, env)
        obs, reward, done, info = env.step(act)
        step += 1
        ev = info.get("events", [])
        if act == env.WAIT_ACTION:
            desc = "WAIT"
        else:
            desc = f"T{tid}->M{mid} dl={dl}"
        if reward != 0 or done:
            evs = " | ".join(ev) if ev else ""
            print(f"  {step:2d} t={obs['current_time']:2d} {desc:16s} r={reward:+6.0f}  {evs}")
    print(f"  TOTAL={info['total_reward']:+.0f}  done={obs['num_completed_tasks']}  miss={obs['num_missed_tasks']}")
    print()

run("EASY", generate_easy)
run("MEDIUM", generate_medium)
run("HARD", generate_hard)
