"""
scheduler_env.py — The RL Environment for Intelligent Task Scheduling.

This is the main file. It implements:
  - reset()  → start a new episode
  - step()   → take one action, get (observation, reward, done, info)
  - state()  → get the current observation (what the agent sees)

Design principles:
  - Deterministic: same config + same actions = same results, always.
  - The agent is NEVER shown: remaining_time, arrival_queue, or finish_time.
  - Time only advances when the agent chooses WAIT.
  - Multiple task assignments can happen at the same time step.
"""

from .models import Task, Machine, PENDING, RUNNING, COMPLETED, MISSED


# ─────────────────────────────────────────────
#  Reward Constants (all in one place for easy tuning)
# ─────────────────────────────────────────────
REWARD_ON_TIME = 10        # Per priority point, for completing before deadline
REWARD_LATE = -5           # Per priority point, for completing after deadline
REWARD_MISSED = -10        # Per priority point, for impossible-to-complete task
REWARD_INVALID = -2        # Flat penalty for an illegal action
REWARD_IDLE = -1           # Per idle machine when WAIT with assignable tasks
REWARD_DEADLOCK = -5       # WAIT when no machines busy but tasks are pending
BONUS_ALL_ON_TIME = 20     # End-of-episode bonus: everything on time
BONUS_ALL_DONE = 5         # End-of-episode bonus: all done, but some late


# ─────────────────────────────────────────────
#  The WAIT action is encoded as the last action index
# ─────────────────────────────────────────────
# If we have N task slots and M machines, then:
#   action 0 to (N*M - 1)  = assign task t to machine m
#   action N*M              = WAIT
# Formula: action_id = task_index * num_machines + machine_index


class SchedulerEnv:
    """
    Reinforcement Learning environment for task scheduling.

    Usage:
        env = SchedulerEnv(task_configs, machine_configs)
        obs = env.reset()
        while not done:
            action = agent.pick_action(obs)
            obs, reward, done, info = env.step(action)
    """

    def __init__(self, task_configs, machine_configs,
                 max_time=50, max_steps=200):
        """
        Args:
            task_configs: list of dicts, each with keys:
                {task_id, duration, deadline, priority,
                 required_machine_type, arrival_time}
            machine_configs: list of dicts, each with keys:
                {machine_id, machine_type}
            max_time: episode ends if current_time exceeds this
            max_steps: episode ends if agent takes more actions than this
        """
        self._task_configs = task_configs          # Saved for reset()
        self._machine_configs = machine_configs    # Saved for reset()
        self._max_time = max_time
        self._max_steps = max_steps

        # These will be set in reset()
        self._tasks = {}                # dict: task_id → Task object
        self._machines = {}             # dict: machine_id → Machine object
        self._arrival_queue = []        # tasks not yet revealed (sorted by arrival)
        self._current_time = 0
        self._step_count = 0
        self._done = False
        self._total_reward = 0.0

        # Dimensions for action encoding
        self._num_task_slots = len(task_configs)
        self._num_machines = len(machine_configs)
        self._task_id_list = []         # ordered list of task IDs (for index→id mapping)

        # The WAIT action is the last one
        self.WAIT_ACTION = self._num_task_slots * self._num_machines

    # ═══════════════════════════════════════════
    #  RESET — Start a fresh episode
    # ═══════════════════════════════════════════
    def reset(self):
        """
        Initialize all internal state and return the first observation.

        Returns:
            observation (dict): what the agent sees at time=0
        """
        # Reset clock and counters
        self._current_time = 0
        self._step_count = 0
        self._done = False
        self._total_reward = 0.0

        # Build the ordered list of task IDs (index 0, 1, 2, ... maps to these)
        self._task_id_list = [tc["task_id"] for tc in self._task_configs]

        # Create all Task objects from config
        all_tasks = []
        for tc in self._task_configs:
            task = Task(
                task_id=tc["task_id"],
                duration=tc["duration"],
                deadline=tc["deadline"],
                priority=tc["priority"],
                required_machine_type=tc["required_machine_type"],
                arrival_time=tc["arrival_time"],
            )
            all_tasks.append(task)

        # Split into: visible now (arrival_time == 0) vs. arrival queue
        self._tasks = {}           # visible tasks (keyed by task_id)
        self._arrival_queue = []   # future tasks (sorted by arrival_time)

        for task in all_tasks:
            if task.arrival_time <= 0:
                # Task is available immediately
                self._tasks[task.task_id] = task
            else:
                # Task arrives later — hidden from agent
                self._arrival_queue.append(task)

        # Sort arrival queue by arrival_time (earliest first)
        self._arrival_queue.sort(key=lambda t: t.arrival_time)

        # Check for tasks that arrive already impossible
        # (deadline already passed or too tight at arrival)
        self._check_impossible_on_arrival()

        # Create Machine objects
        self._machines = {}
        for mc in self._machine_configs:
            machine = Machine(
                machine_id=mc["machine_id"],
                machine_type=mc["machine_type"],
            )
            self._machines[machine.machine_id] = machine

        return self.state()

    # ═══════════════════════════════════════════
    #  STATE — What the agent sees (observation)
    # ═══════════════════════════════════════════
    def state(self):
        """
        Build the observation dict.

        This is everything the agent is allowed to see.
        It deliberately EXCLUDES:
          - remaining_time on machines
          - which task is on which machine
          - the arrival queue (future tasks)
          - start_time and finish_time of tasks
        """
        # Count tasks by status (only visible tasks)
        num_pending = sum(1 for t in self._tasks.values() if t.status == PENDING)
        num_active = sum(1 for t in self._tasks.values() if t.status == RUNNING)
        num_completed = sum(1 for t in self._tasks.values() if t.status == COMPLETED)
        num_missed = sum(1 for t in self._tasks.values() if t.status == MISSED)

        # Build task observation list (one entry per task slot)
        # Tasks not yet arrived show as empty dicts (all zeros)
        task_obs = []
        for task_id in self._task_id_list:
            if task_id in self._tasks:
                task_obs.append(self._tasks[task_id].to_obs())
            else:
                # Slot is empty — task hasn't arrived yet
                # All zeros tell the agent "nothing here yet"
                task_obs.append({
                    "task_id": 0,
                    "duration": 0,
                    "deadline": 0,
                    "priority": 0,
                    "required_machine_type": 0,
                    "status": 0,
                })

        # Build machine observation list
        machine_obs = [m.to_obs() for m in self._machines.values()]

        return {
            "current_time": self._current_time,
            "num_pending_tasks": num_pending,
            "num_active_tasks": num_active,
            "num_completed_tasks": num_completed,
            "num_missed_tasks": num_missed,
            "tasks": task_obs,
            "machines": machine_obs,
        }

    # ═══════════════════════════════════════════
    #  STEP — Take one action
    # ═══════════════════════════════════════════
    def step(self, action):
        """
        Process one agent action.

        Args:
            action (int): Either a (task, machine) pair encoded as
                          task_index * num_machines + machine_index,
                          or WAIT_ACTION.

        Returns:
            observation (dict): updated state after action
            reward (float): reward earned this step
            done (bool): whether the episode has ended
            info (dict): extra information for debugging
        """
        # If episode already over, don't process anything
        if self._done:
            return self.state(), 0.0, True, {"error": "Episode already done"}

        self._step_count += 1
        reward = 0.0
        info = {}

        # ── Decide: is this WAIT or an assignment? ──
        if action == self.WAIT_ACTION:
            reward, info = self._handle_wait()
        else:
            reward, info = self._handle_assign(action)

        # ── Check if episode should end ──
        self._total_reward += reward
        terminated, truncated = self._check_done()

        if terminated or truncated:
            self._done = True
            # Add episode-end bonus
            bonus = self._compute_episode_bonus()
            reward += bonus
            self._total_reward += bonus
            info["episode_bonus"] = bonus
            info["terminated"] = terminated
            info["truncated"] = truncated

        info["step"] = self._step_count
        info["total_reward"] = self._total_reward

        return self.state(), reward, self._done, info

    # ═══════════════════════════════════════════
    #  ASSIGN — Handle a (task, machine) action
    # ═══════════════════════════════════════════
    def _handle_assign(self, action):
        """
        Attempt to assign a task to a machine.

        Validates everything first. If invalid, return penalty.
        If valid, update task and machine state. Do NOT advance time.
        """
        # Decode the action integer back to (task_index, machine_index)
        task_index = action // self._num_machines
        machine_index = action % self._num_machines

        # ── Validation step 1: indices in range? ──
        if task_index < 0 or task_index >= self._num_task_slots:
            return REWARD_INVALID, {"error": "Task index out of range"}

        if machine_index < 0 or machine_index >= self._num_machines:
            return REWARD_INVALID, {"error": "Machine index out of range"}

        # ── Get the actual task and machine ──
        task_id = self._task_id_list[task_index]
        machine_id = list(self._machines.keys())[machine_index]

        # ── Validation step 2: task exists and is visible? ──
        if task_id not in self._tasks:
            return REWARD_INVALID, {"error": f"Task {task_id} not yet arrived"}

        task = self._tasks[task_id]
        machine = self._machines[machine_id]

        # ── Validation step 3: task is pending? ──
        if task.status != PENDING:
            status_name = {0: "PENDING", 1: "RUNNING", 2: "COMPLETED", 3: "MISSED"}
            return REWARD_INVALID, {
                "error": f"Task {task_id} is {status_name.get(task.status, '?')}, not PENDING"
            }

        # ── Validation step 4: machine is idle? ──
        if machine.is_busy:
            return REWARD_INVALID, {
                "error": f"Machine {machine_id} is busy"
            }

        # ── Validation step 5: machine type matches? ──
        if machine.machine_type != task.required_machine_type:
            return REWARD_INVALID, {
                "error": (f"Machine {machine_id} is type {machine.machine_type}, "
                          f"but task {task_id} needs type {task.required_machine_type}")
            }

        # ═══ All checks passed — assign the task ═══

        # Update task state
        task.status = RUNNING
        task.start_time = self._current_time
        task.finish_time = self._current_time + task.duration

        # Update machine state (remaining_time is HIDDEN from agent)
        machine.is_busy = True
        machine.assigned_task_id = task.task_id
        machine.remaining_time = task.duration

        # Time does NOT advance — agent can assign more tasks this tick
        return 0.0, {
            "assigned": f"Task {task_id} → Machine {machine_id}",
            "will_finish_at": task.finish_time,
        }

    # ═══════════════════════════════════════════
    #  WAIT — Advance time to next event
    # ═══════════════════════════════════════════
    def _handle_wait(self):
        """
        Advance time to the next event (machine completion or task arrival).

        This is where most of the environment's logic lives:
        1. Find the next event time
        2. Jump the clock forward
        3. Reveal any newly arriving tasks
        4. Complete any finished tasks
        5. Mark any newly impossible tasks as missed
        """
        reward = 0.0
        info = {"action": "WAIT", "events": []}

        # ── Step 1: Find the next event time ──

        # Candidate A: earliest machine completion
        busy_machines = [m for m in self._machines.values() if m.is_busy]
        next_completion = None
        if busy_machines:
            # The machine with the least remaining_time finishes first
            next_completion = self._current_time + min(m.remaining_time for m in busy_machines)

        # Candidate B: earliest task arrival
        next_arrival = None
        if self._arrival_queue:
            next_arrival = self._arrival_queue[0].arrival_time

        # ── Step 2: Handle edge cases ──

        # Deadlock: nothing is busy, but pending tasks exist
        pending_tasks = [t for t in self._tasks.values() if t.status == PENDING]

        if not busy_machines and pending_tasks and next_arrival is None:
            # No machines running, tasks are waiting, nothing arriving
            # → Agent should have assigned instead of waiting. Penalty.
            self._current_time += 1  # Force time forward to prevent infinite loop
            info["events"].append("DEADLOCK: forced time +1")
            return REWARD_DEADLOCK, info

        # Nothing happening at all and no arrivals → episode should end
        if not busy_machines and not pending_tasks and next_arrival is None:
            info["events"].append("No work remaining")
            return 0.0, info

        # ── Idle penalty: did the agent WAIT when it could have assigned? ──
        # Count machines that are idle AND have a compatible pending task
        idle_penalty = 0
        for machine in self._machines.values():
            if not machine.is_busy:
                # Check if any pending task could go on this machine
                for task in self._tasks.values():
                    if (task.status == PENDING and
                            task.required_machine_type == machine.machine_type):
                        idle_penalty += REWARD_IDLE
                        break  # One penalty per idle machine, not per task
        reward += idle_penalty

        # ── Step 3: Jump to the next event ──
        # Pick whichever comes first: completion or arrival
        candidates = []
        if next_completion is not None:
            candidates.append(next_completion)
        if next_arrival is not None:
            candidates.append(next_arrival)

        if not candidates:
            # Shouldn't reach here (handled above), but be safe
            return reward, info

        next_time = min(candidates)
        time_delta = next_time - self._current_time

        # Sanity check: time should always move forward
        if time_delta <= 0:
            # This can happen if arrival_time == current_time, which means
            # the task should have already been revealed. Handle gracefully.
            time_delta = max(time_delta, 0)

        self._current_time = next_time
        info["time_advanced_to"] = self._current_time

        # ── Step 4: Reveal newly arrived tasks ──
        # Move tasks from arrival_queue → visible tasks
        newly_arrived = []
        while (self._arrival_queue and
               self._arrival_queue[0].arrival_time <= self._current_time):
            task = self._arrival_queue.pop(0)
            task.status = PENDING
            self._tasks[task.task_id] = task
            newly_arrived.append(task.task_id)

            # Edge case: did this task arrive already impossible?
            # (deadline already passed, or not enough time)
            if self._current_time > task.deadline:
                # Deadline already in the past
                task.status = MISSED
                reward += REWARD_MISSED * task.priority
                info["events"].append(f"Task {task.task_id} arrived ALREADY MISSED (deadline passed)")
            elif (task.deadline - self._current_time) < task.duration:
                # Not enough time left even if started immediately
                task.status = MISSED
                reward += REWARD_MISSED * task.priority
                info["events"].append(f"Task {task.task_id} arrived IMPOSSIBLE (too tight)")

        if newly_arrived:
            info["events"].append(f"Tasks arrived: {newly_arrived}")

        # ── Step 5: Complete finished tasks ──
        for machine in self._machines.values():
            if machine.is_busy:
                # Subtract the time that passed
                machine.remaining_time -= time_delta

                if machine.remaining_time <= 0:
                    # Task is done!
                    machine.remaining_time = 0
                    machine.is_busy = False

                    # Find the task that was running
                    task = self._tasks[machine.assigned_task_id]
                    task.status = COMPLETED

                    # Reward depends on whether it met the deadline
                    if task.finish_time <= task.deadline:
                        # On time! 
                        r = REWARD_ON_TIME * task.priority
                        reward += r
                        info["events"].append(
                            f"Task {task.task_id} COMPLETED on time (reward: +{r})"
                        )
                    else:
                        # Late completion
                        r = REWARD_LATE * task.priority
                        reward += r
                        info["events"].append(
                            f"Task {task.task_id} COMPLETED LATE (penalty: {r})"
                        )

                    machine.assigned_task_id = None

        # ── Step 6: Mark newly impossible pending tasks ──
        # A pending task is "missed" if there's not enough time left
        # to finish it even if started right now
        for task in self._tasks.values():
            if task.status == PENDING:
                if self._current_time > (task.deadline - task.duration):
                    task.status = MISSED
                    r = REWARD_MISSED * task.priority
                    reward += r
                    info["events"].append(
                        f"Task {task.task_id} MISSED (can no longer finish by deadline {task.deadline})"
                    )

        return reward, info

    # ═══════════════════════════════════════════
    #  DONE CHECK — Is the episode over?
    # ═══════════════════════════════════════════
    def _check_done(self):
        """
        Check termination conditions.

        Returns:
            terminated (bool): True if all tasks are resolved naturally
            truncated (bool): True if hit time or step limit
        """
        # ── Truncation: time or step limit ──
        if self._current_time > self._max_time:
            return False, True    # truncated

        if self._step_count >= self._max_steps:
            return False, True    # truncated

        # ── Termination: all tasks resolved ──
        # All tasks must be revealed (arrival queue empty)
        # AND every visible task must be completed or missed
        if self._arrival_queue:
            return False, False   # Still have unrevealed tasks

        all_resolved = all(
            t.status in (COMPLETED, MISSED)
            for t in self._tasks.values()
        )

        if all_resolved:
            return True, False    # terminated naturally

        return False, False       # Episode continues

    # ═══════════════════════════════════════════
    #  EPISODE BONUS — Extra reward at the end
    # ═══════════════════════════════════════════
    def _compute_episode_bonus(self):
        """
        Calculate the end-of-episode bonus.

        +20 if ALL tasks completed on time
        +5  if all tasks completed (but some were late)
         0  if any tasks were missed entirely
        """
        all_tasks = list(self._tasks.values())

        if not all_tasks:
            return 0

        # Check: are there any missed tasks?
        any_missed = any(t.status == MISSED for t in all_tasks)
        if any_missed:
            return 0

        # All tasks completed (no misses). Were any late?
        any_late = any(
            t.status == COMPLETED and t.finish_time > t.deadline
            for t in all_tasks
        )

        if any_late:
            return BONUS_ALL_DONE        # +5
        else:
            return BONUS_ALL_ON_TIME     # +20

    # ═══════════════════════════════════════════
    #  HELPER — Check impossible tasks on arrival
    # ═══════════════════════════════════════════
    def _check_impossible_on_arrival(self):
        """
        At reset, check if any initially visible tasks are already
        impossible to complete (deadline too tight).
        """
        for task in list(self._tasks.values()):
            if task.deadline <= self._current_time:
                task.status = MISSED
            elif (task.deadline - self._current_time) < task.duration:
                task.status = MISSED

    # ═══════════════════════════════════════════
    #  DEBUG — Print full internal state
    # ═══════════════════════════════════════════
    def debug_state(self):
        """
        Print EVERYTHING, including hidden state.
        Only use this for debugging — never expose to the agent!
        """
        print(f"\n{'='*50}")
        print(f"  TIME: {self._current_time}  |  STEP: {self._step_count}  "
              f"|  REWARD: {self._total_reward:.1f}  |  DONE: {self._done}")
        print(f"{'='*50}")

        print("\n  TASKS (visible):")
        for t in self._tasks.values():
            print(f"    {t}")

        if self._arrival_queue:
            print(f"\n  ARRIVAL QUEUE (hidden, {len(self._arrival_queue)} tasks):")
            for t in self._arrival_queue:
                print(f"    {t}")
        else:
            print("\n  ARRIVAL QUEUE: empty")

        print("\n  MACHINES:")
        for m in self._machines.values():
            print(f"    {m}")
        print()
