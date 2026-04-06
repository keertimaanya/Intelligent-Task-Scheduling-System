"""
scenarios.py — Task & machine configs for 3 difficulty levels.

Each function returns (task_configs, machine_configs) ready to plug
straight into SchedulerEnv.

Difficulty progression:

  EASY    → Everything fits. Plenty of time. One machine type.
            Agent just needs to learn: "assign tasks, then wait."

  MEDIUM  → Mixed priorities. Tighter deadlines. Two machine types.
            Agent must learn: "do high-priority first" and "match types."

  HARD    → Dynamic arrivals. Conflicts (more tasks than machines).
            Overlapping deadlines that force trade-offs.
            Agent must learn: "sacrifice low-priority to save high-priority."
"""


# ═══════════════════════════════════════════════
#  EASY — The Training Wheels
# ═══════════════════════════════════════════════
#
#  What makes it easy:
#    ✓ All tasks arrive at time 0 (no surprises)
#    ✓ Only 1 machine type (no type matching needed)
#    ✓ Generous deadlines (plenty of slack)
#    ✓ 3 tasks, 2 machines (more capacity than work)
#    ✓ All same priority (no triage decisions)
#
#  What the agent learns here:
#    - Basic mechanics: assign → wait → collect reward
#    - That WAIT advances time
#    - That invalid actions get penalized

def generate_easy():
    """
    3 tasks, 2 machines, all type 1, all arrive at time 0.
    Every task can finish well before its deadline.

    Timeline if optimal:
      time 0: Assign T1→M1, T2→M2
      time 2: T1 done, assign T3→M1
      time 3: T2 done
      time 4: T3 done
      All deadlines met easily. Total reward = +50 (3×10 + 20 bonus)
    """
    task_configs = [
        {
            "task_id": 1,
            "duration": 2,          # Takes 2 time units
            "deadline": 8,          # Due at time 8 — lots of room
            "priority": 1,          # All same priority
            "required_machine_type": 1,
            "arrival_time": 0,      # Available immediately
        },
        {
            "task_id": 2,
            "duration": 3,
            "deadline": 10,
            "priority": 1,
            "required_machine_type": 1,
            "arrival_time": 0,
        },
        {
            "task_id": 3,
            "duration": 2,
            "deadline": 10,
            "priority": 1,
            "required_machine_type": 1,
            "arrival_time": 0,
        },
    ]

    # 2 machines, both type 1 — any task can go on any machine
    machine_configs = [
        {"machine_id": 1, "machine_type": 1},
        {"machine_id": 2, "machine_type": 1},
    ]

    return task_configs, machine_configs


# ═══════════════════════════════════════════════
#  MEDIUM — Now You Have to Think
# ═══════════════════════════════════════════════
#
#  What's new compared to EASY:
#    ✗ Two machine types — must match task to correct machine
#    ✗ Mixed priorities (1, 2, 3) — order matters now
#    ✗ Tighter deadlines — less slack, some tasks could be missed
#      if the agent assigns in a bad order
#    ✗ 5 tasks, 2 machines — more work than capacity
#
#  What the agent learns here:
#    - Type matching: task type must match machine type
#    - Priority triage: do ⭐⭐⭐ before ⭐ when time is tight
#    - Sequencing: the ORDER of assignments matters
#
#  Key tension: T2 (high priority, tight deadline) vs T3 (low priority,
#  easy deadline). Both need machine type 1. Who goes first?

def generate_medium():
    """
    5 tasks, 2 machines (type 1 and type 2), mixed priorities.
    Deadlines are tight but all tasks CAN be completed if ordered well.

    The trap: If agent does T3 before T2 on M1, T2 will miss its deadline.
    Correct order: T2 first (high priority, tight), then T3 (low, relaxed).
    """
    task_configs = [
        {
            "task_id": 1,
            "duration": 2,
            "deadline": 5,          # Moderate deadline
            "priority": 2,          # ⭐⭐ Medium
            "required_machine_type": 1,   # Needs machine type 1
            "arrival_time": 0,
        },
        {
            "task_id": 2,
            "duration": 3,
            "deadline": 6,          # TIGHT — only 3 units of slack from time 0
            "priority": 3,          # ⭐⭐⭐ HIGH — must do this first!
            "required_machine_type": 1,   # Also needs type 1  ← conflict with T1!
            "arrival_time": 0,
        },
        {
            "task_id": 3,
            "duration": 2,
            "deadline": 12,         # Very relaxed deadline
            "priority": 1,          # ⭐ Low
            "required_machine_type": 1,   # Also type 1 — three tasks sharing one machine!
            "arrival_time": 0,
        },
        {
            "task_id": 4,
            "duration": 2,
            "deadline": 4,          # Tight deadline
            "priority": 2,          # ⭐⭐ Medium
            "required_machine_type": 2,   # Needs machine type 2
            "arrival_time": 0,
        },
        {
            "task_id": 5,
            "duration": 3,
            "deadline": 8,          # Moderate
            "priority": 1,          # ⭐ Low
            "required_machine_type": 2,   # Also type 2
            "arrival_time": 0,
        },
    ]

    # Only 2 machines — one of each type
    # Type 1 has 3 tasks competing for it!
    machine_configs = [
        {"machine_id": 1, "machine_type": 1},   # Handles T1, T2, T3
        {"machine_id": 2, "machine_type": 2},   # Handles T4, T5
    ]

    return task_configs, machine_configs


# ═══════════════════════════════════════════════
#  HARD — Welcome to the Real World
# ═══════════════════════════════════════════════
#
#  What's new compared to MEDIUM:
#    ✗ Dynamic arrivals — tasks appear mid-episode
#    ✗ Impossible to complete ALL tasks — forced trade-offs
#    ✗ Deadline conflicts — two high-priority tasks compete
#      for the same machine at the same time
#    ✗ 7 tasks, 3 machines — but arrival timing creates bottlenecks
#
#  What the agent learns here:
#    - Sacrifice: sometimes you MUST let a low-priority task miss
#      to save a high-priority one
#    - Capacity reservation: don't fill all machines right away,
#      because surprises are coming
#    - Urgency vs priority: a low-priority task with an imminent
#      deadline vs a high-priority task with a later deadline —
#      which one first?
#
#  Design: It's IMPOSSIBLE to get all 7 tasks on time. The agent
#  must learn which ones to sacrifice for maximum total reward.

def generate_hard():
    """
    7 tasks, 3 machines, dynamic arrivals, forced conflicts.

    Wave 1 (time 0): 3 tasks arrive — fills all 3 machines
    Wave 2 (time 2): 2 tasks arrive — but machines are still busy!
                     Agent must triage.
    Wave 3 (time 5): 2 more tasks with very tight deadlines.
                     One will almost certainly be missed.

    Key conflict at time 2:
      T4 (⭐⭐⭐, deadline 5, duration 2, type 1) — urgent AND important
      T5 (⭐, deadline 6, duration 3, type 1) — can probably wait
      But M1 is still busy with T1 until time 3!
      → T4 will start at time 3, finish at 5 — just barely on time
      → T5 must wait even longer — might miss

    Key conflict at time 5:
      T6 and T7 both need type 2, but there's only one type-2 machine.
      T6 is ⭐⭐⭐ (deadline 8), T7 is ⭐⭐ (deadline 7).
      T7 has the earlier deadline but lower priority.
      What does the agent choose? (Hint: T6 gives more reward)
    """
    task_configs = [
        # ── Wave 1: Available immediately ──
        {
            "task_id": 1,
            "duration": 3,
            "deadline": 8,
            "priority": 2,          # ⭐⭐ Medium
            "required_machine_type": 1,
            "arrival_time": 0,
        },
        {
            "task_id": 2,
            "duration": 2,
            "deadline": 5,
            "priority": 1,          # ⭐ Low
            "required_machine_type": 2,
            "arrival_time": 0,
        },
        {
            "task_id": 3,
            "duration": 2,
            "deadline": 4,
            "priority": 2,          # ⭐⭐ Medium
            "required_machine_type": 3,
            "arrival_time": 0,
        },

        # ── Wave 2: Arrives at time 2 (surprise!) ──
        {
            "task_id": 4,
            "duration": 2,
            "deadline": 5,          # VERY tight — must start by time 3
            "priority": 3,          # ⭐⭐⭐ HIGH
            "required_machine_type": 1,   # Conflicts with T1 on M1!
            "arrival_time": 2,
        },
        {
            "task_id": 5,
            "duration": 3,
            "deadline": 9,
            "priority": 1,          # ⭐ Low — sacrifice candidate
            "required_machine_type": 1,
            "arrival_time": 2,
        },

        # ── Wave 3: Arrives at time 5 (another surprise!) ──
        {
            "task_id": 6,
            "duration": 2,
            "deadline": 8,
            "priority": 3,          # ⭐⭐⭐ HIGH
            "required_machine_type": 2,   # Conflicts with T7!
            "arrival_time": 5,
        },
        {
            "task_id": 7,
            "duration": 2,
            "deadline": 7,          # Earlier deadline but lower priority
            "priority": 2,          # ⭐⭐ Medium
            "required_machine_type": 2,   # Same type as T6!
            "arrival_time": 5,
        },
    ]

    # 3 machines, one per type
    machine_configs = [
        {"machine_id": 1, "machine_type": 1},   # Handles T1, T4, T5
        {"machine_id": 2, "machine_type": 2},   # Handles T2, T6, T7
        {"machine_id": 3, "machine_type": 3},   # Handles T3 only
    ]

    return task_configs, machine_configs


# ═══════════════════════════════════════════════
#  Summary: How Difficulty Scales
# ═══════════════════════════════════════════════
#
#  ┌──────────┬───────┬────────┬────────┐
#  │          │ EASY  │ MEDIUM │ HARD   │
#  ├──────────┼───────┼────────┼────────┤
#  │ Tasks    │   3   │   5    │   7    │
#  │ Machines │   2   │   2    │   3    │
#  │ Types    │   1   │   2    │   3    │
#  │ Arrivals │ all@0 │ all@0  │ 3 waves│
#  │ Slack    │ lots  │ tight  │ none   │
#  │ Priority │ same  │ mixed  │ conflicts│
#  │ Solvable │ 100%  │ 100%   │ NOT all│
#  └──────────┴───────┴────────┴────────┘
