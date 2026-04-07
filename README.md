---
title: Intelligent Task Scheduling
emoji: 🗓️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
tags:
  - openenv
---

# Intelligent Task Scheduling System (OpenEnv)

A deterministic, event-driven RL environment for dynamic task scheduling, built to the **OpenEnv** benchmark specification. Agents must assign tasks to machines before deadlines expire, balancing priority trade-offs across three difficulty levels.

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

### Observation Space
The Environment State (`Observation`) is returned as a heavily typed Pydantic JSON schema:
```json
{
  "current_time": 2,
  "tasks": [{"id": 1, "duration": 2, "deadline": 8, "priority": 1, "status": 0}],
  "machines": [{"id": 0, "is_busy": 0, "busy_until": 0}]
}
```
*Note: Future "surprise" tasks are mathematically hidden from the observation space to prevent agent cheating.*

### Action Space

The environment uses a discrete integer action space.

Each action represents assigning a task to a machine and is encoded as:

Action = task_index * num_machines + machine_index

This compact encoding allows efficient representation of all possible task-machine assignments in a single discrete space.

A special WAIT action is also provided, allowing the agent to skip a timestep when no beneficial assignment is available.

---

## Project Structure
```
.
├── openenv.yaml            # OpenEnv specification file
├── inference.py            # LLM baseline inference script (root, required)
├── Dockerfile              # HF Spaces compatible (port 7860)
├── docker-compose.yml
├── requirements.txt
├── README.md
│
├── api/
│   └── server.py           # FastAPI server (reset, step, state, grade, tasks)
│
├── env/
│   ├── scheduler_env.py    # Core RL environment (reset, step, state)
│   ├── models.py           # Task & Machine data classes
│   ├── scenarios.py        # Easy / Medium / Hard task generators
│   └── __init__.py
│
├── agents/
│   └── baseline_edf.py     # Earliest Deadline First heuristic agent
│
├── graders/
│   └── grader.py           # Scoring: WCR, PWE, ER, AA -> 0.0-1.0
│
└── tests/
    └── ...
```

---

## Quick Start

### 1. Run the Environment Server

**Docker (recommended):**
```bash
docker-compose up --build
```

**Local:**
```bash
pip install -r requirements.txt
uvicorn api.server:app --host 0.0.0.0 --port 7860
```

The server runs at `http://localhost:7860`. API docs at `http://localhost:7860/docs`.

### 2. Run the Baseline Inference

Set the required environment variables and run:

```bash
# Linux / Mac
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="sk-your-openai-key"
export ENV_URL="http://localhost:7860"
python inference.py

# Windows PowerShell
$env:API_BASE_URL="https://api.openai.com/v1"
$env:MODEL_NAME="gpt-4o-mini"
$env:HF_TOKEN="sk-your-openai-key"
$env:ENV_URL="http://localhost:7860"
python inference.py
```

### 3. Run the EDF Heuristic Baseline
Run the mathematical deterministic baseline to test environment physics:
```bash
python agents/baseline_edf.py
```
**Reproducible Baseline Scores:**
* `easy`: **0.95** (Tested via Heuristic, +50 raw total reward)
* `medium`: **1.00** (Tested via Heuristic, +110 raw total reward)
* `hard`: **0.78** (Tested via Heuristic, +80 raw total reward, 1 intentional task sacrifice required)

---

## API Endpoints

| Method | Path     | Description                                |
|--------|----------|--------------------------------------------|
| GET    | `/`      | Health check (returns 200)                 |
| POST   | `/reset` | Start a new episode `{"level": "easy"}`    |
| POST   | `/step`  | Take an action `{"action": 0}`             |
| GET    | `/state` | Get current observation                    |
| GET    | `/tasks` | List all 3 task definitions                |
| POST   | `/grade` | Grade a completed episode (returns 0.0-1.0)|

---

## Environment Variables (Mandatory for Submission)

| Variable       | Description                          |
|----------------|--------------------------------------|
| `API_BASE_URL` | The API endpoint for the LLM         |
| `MODEL_NAME`   | The model identifier for inference   |
| `HF_TOKEN`     | Your Hugging Face / API key          |

---

## Inference Output Format

The `inference.py` script emits strictly formatted logs matching the OpenEnv STDOUT constraints:

```
[START] task=easy env=intelligent-task-scheduling model=gpt-4o-mini
[STEP] step=1 action=ASSIGN(0) reward=0.00 done=false error=null
[STEP] step=2 action=WAIT reward=10.00 done=false error=null
...
[END] success=true steps=6 score=0.95 rewards=0.00,10.00,10.00,30.00
```

---

## Grading

Each task is scored between **0.0** and **1.0** using four weighted metrics:

- **WCR** — Weighted Completion Rate (did you complete high-priority tasks?)
- **PWE** — Priority-Weighted Earliness (how much deadline buffer remained?)
- **ER**  — Efficiency Ratio (actual reward vs theoretical maximum)
- **AA**  — Action Accuracy (ratio of valid actions)

---

## Tasks

| ID     | Tasks | Machines | Key Challenge                        |
|--------|-------|----------|--------------------------------------|
| easy   | 3     | 2        | Simple scheduling, generous deadlines|
| medium | 5     | 2        | Type matching, mixed priorities      |
| hard   | 7     | 3        | Dynamic arrivals, forced trade-offs  |
