# Intelligent Task Scheduling System (OpenEnv)

A deterministic, event-driven Reinforcement Learning (RL) environment tailored as a competitive benchmark for dynamic task scheduling. Built under the OpenEnv benchmark structure, it challenges agents to manage task queues, machine capabilities, and deadlines across three distinct difficulty levels.

---

##  The Challenge
You act as a Task Scheduler in a dynamic environment (think cloud resource manager, or an automated factory). 
The goal is to map arriving tasks to available machines before their deadlines expire. 
Tasks carry varying **priorities**, meaning missing a high-priority task penalizes you significantly more than missing a low-priority task.

### Why is this hard?
- **Future arrives unannounced**: Higher difficulty levels feature "surpris" task arrivals mid-episode. You must decide whether to save a machine for a potential high-priority drop or fill it immediately.
- **Forced Trade-offs**: In the `HARD` setting, the sheer volume and tightness of deadlines guarantees you **cannot** complete everything. You must sacrifice low-priority tasks to survive.

---

## Environment Design

### Action Space

The environment uses a discrete integer action space.

Each action represents assigning a task to a machine and is encoded as:

Action = task_index * num_machines + machine_index

This compact encoding allows efficient representation of all possible task-machine assignments in a single discrete space.

A special WAIT action is also provided, allowing the agent to skip a timestep when no beneficial assignment is available.

---

##  Project Structure

```
├── README.md               # You are here
├── requirements.txt        # Core requirements (FastAPI, pydantic)
├── Dockerfile              # Dockerized environment
├── docker-compose.yml      
│
├── api/                    
│   └── server.py           # REST API exposing the Environment
│
├── env/                    
│   ├── scheduler_env.py    # Core RL Environment logic (reset, step, state)
│   ├── models.py           # Core schemas (Task, Machine definitions)
│   ├── scenarios.py        # Maps the EASY, MEDIUM, HARD configurations
│   └── __init__.py         
│
├── agents/                 
│   └── baseline_edf.py     # Greedy Earliest Deadline First (EDF) agent
│
├── graders/                
│   └── grader.py           # Mathematical normalized evaluating logic (0.0 - 1.0)
│
└── tests/                  # End-to-end tests for the API and Environment
```

---

##  Running the Environment

### Method 1: Docker (Recommended)
You can instantly spin up the isolated environment and its REST API natively.
```bash
docker-compose up --build
```
*The FastAPI server will be available at `http://localhost:8000`.*

### Method 2: Local Python
Install the minimal dependencies and run `uvicorn`.
```bash
pip install -r requirements.txt
uvicorn api.server:app --host 0.0.0.0 --port 8000
```
*API documentation is auto-generated at `http://localhost:8000/docs`.*

---

##  Interaction via API (REST interface)

Agents interface with the OpenEnv server via three primary endpoints.

#### 1. Start an Episode
```bash
POST /reset
Body: {"level": "hard"}
```
*Returns the initial State observation where `current_time` is zero.*

#### 2. Get Current State observation
```bash
GET /state
```
*Returns the observation exactly as is without modifying or ticking the clock.*

#### 3. Step the Environment
```bash
POST /step
Body: {"action": 8} 
```
*Actions are integer encoded (`task_ID * num_machines + machine_ID`). The highest integer corresponds to the `WAIT` action. It returns `observation`, `reward` (float), `done` (bool), and `info` dict.*

---

##  Baseline Agent (EDF)
A simple Earliest Deadline First (EDF) agent is provided. It serves as a benchmark that strictly prioritizes urgency over importance.
It easily conquers `EASY` and `MEDIUM` but naturally fails on `HARD` because it cannot execute strategic sacrifices on competing priorities.

**To run the baseline:**
```bash
python agents/baseline_edf.py
```

---

##  Graders
Performance is mathematically mapped mathematically between `0.0` (Worst) and `1.0` (Perfect) across 4 metrics:
1. **WCR (Weighted Completion Rate)**: Normalizes on-time completions by priority.
2. **PWE (Priority-Weighted Earliness)**: Judges how much safe "slack" or buffer remained when completing.
3. **ER (Efficiency Ratio)**: Tracks total accumulated reward vs the mathematically possible optimal maximum per-level.
4. **AA (Action Accuracy)**: Checks decision validity count (protects against brute-forcing invalid steps).

The script is available in `graders/grader.py`. It is invoked statically at the end of the episode passing the terminal environment trajectory.
