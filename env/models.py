"""
models.py — Data classes for the Task Scheduling RL Environment.

These are simple containers. No logic lives here — just structure.
Think of them as "forms" that hold information about tasks and machines.
"""


# ─────────────────────────────────────────────
#  Task Status Constants
# ─────────────────────────────────────────────
# We use plain integers so the observation is always numeric.
# Using named constants makes the code readable without needing enums.

PENDING = 0      # Task is waiting to be assigned
RUNNING = 1      # Task is currently executing on a machine
COMPLETED = 2    # Task finished successfully
MISSED = 3       # Task can no longer meet its deadline


class Task:
    """
    Represents a single schedulable task.

    Some fields are visible to the agent (observation), others are hidden
    (internal state). This separation is the core of what makes the
    problem an RL problem rather than a planning problem.
    """

    def __init__(self, task_id, duration, deadline, priority,
                 required_machine_type, arrival_time):
        # ── Visible to agent (part of observation) ──
        self.task_id = task_id                           # Unique ID (1, 2, 3, ...)
        self.duration = duration                         # How many time units to complete
        self.deadline = deadline                         # Must finish by this time
        self.priority = priority                         # 1=low, 2=medium, 3=high
        self.required_machine_type = required_machine_type  # Which machine type can run this
        self.status = PENDING                            # Current status (see constants above)

        # ── Hidden from agent (internal state) ──
        self.arrival_time = arrival_time   # When this task becomes visible
        self.start_time = None             # When it was assigned to a machine (None = not started)
        self.finish_time = None            # When it will complete (None = not started)

    def to_obs(self):
        """
        Convert to the observation format the agent sees.
        Returns a dict of ONLY the visible fields.
        Notice: arrival_time, start_time, finish_time are NOT included.
        """
        return {
            "task_id": self.task_id,
            "duration": self.duration,
            "deadline": self.deadline,
            "priority": self.priority,
            "required_machine_type": self.required_machine_type,
            "status": self.status,
        }

    def __repr__(self):
        status_names = {0: "PENDING", 1: "RUNNING", 2: "COMPLETED", 3: "MISSED"}
        return (f"Task(id={self.task_id}, dur={self.duration}, "
                f"dl={self.deadline}, pri={self.priority}, "
                f"type={self.required_machine_type}, "
                f"status={status_names.get(self.status, '?')}, "
                f"arrives={self.arrival_time})")


class Machine:
    """
    Represents a single machine that can execute tasks.

    The agent sees: machine_id, machine_type, is_busy.
    The agent does NOT see: remaining_time, assigned_task_id.
    """

    def __init__(self, machine_id, machine_type):
        # ── Visible to agent ──
        self.machine_id = machine_id       # Unique ID (1, 2, ...)
        self.machine_type = machine_type   # What kind of tasks it can run

        # ── Hidden from agent ──
        self.is_busy = False               # Agent sees this as 0/1
        self.remaining_time = 0            # Hidden: time left on current task
        self.assigned_task_id = None       # Hidden: which task is running

    def to_obs(self):
        """
        Convert to the observation format the agent sees.
        Notice: remaining_time and assigned_task_id are NOT included.
        """
        return {
            "machine_id": self.machine_id,
            "machine_type": self.machine_type,
            "is_busy": int(self.is_busy),   # Convert bool → 0/1 for numeric obs
        }

    def __repr__(self):
        if self.is_busy:
            return (f"Machine(id={self.machine_id}, type={self.machine_type}, "
                    f"BUSY, task={self.assigned_task_id}, "
                    f"remaining={self.remaining_time})")
        return f"Machine(id={self.machine_id}, type={self.machine_type}, IDLE)"
