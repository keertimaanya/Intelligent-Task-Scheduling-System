import os
import sys

# Add the root directory to the python path so we can import from `env`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

from env.scheduler_env import SchedulerEnv
from env.scenarios import generate_easy, generate_medium, generate_hard
from graders.grader import Grader

# ─────────────────────────────────────────────
#  Pydantic Models (typed for OpenEnv compliance)
# ─────────────────────────────────────────────

class ResetRequest(BaseModel):
    level: str = Field(
        default="easy",
        description="Difficulty level: 'easy', 'medium', or 'hard'."
    )

class ActionRequest(BaseModel):
    action: int = Field(
        ...,
        description="Action ID: 0..(N*M-1) for task assignments, N*M for WAIT."
    )

class TaskInfo(BaseModel):
    task_id: int
    status: int
    deadline: int
    priority: int
    required_machine_type: int

class MachineInfo(BaseModel):
    machine_id: int
    machine_type: int
    is_busy: int

class ObservationResponse(BaseModel):
    current_time: int
    num_pending_tasks: int
    num_active_tasks: int
    num_completed_tasks: int
    num_missed_tasks: int
    tasks: List[Dict[str, Any]]
    machines: List[Dict[str, Any]]

class ResetResponse(BaseModel):
    observation: Dict[str, Any]
    wait_action: int

class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]

class StateResponse(BaseModel):
    observation: Dict[str, Any]
    done: bool
    wait_action: int

class TaskDefinition(BaseModel):
    id: str
    name: str
    description: str
    max_reward: float

class GradeResult(BaseModel):
    task_id: str
    score: float
    details: Dict[str, Any]

# ─────────────────────────────────────────────
#  App & State
# ─────────────────────────────────────────────

app = FastAPI(
    title="Intelligent Task Scheduling OpenEnv",
    description="OpenEnv-compliant RL environment for dynamic task scheduling.",
    version="1.0.0",
)

# Global environment instance
env: Optional[SchedulerEnv] = None
current_level: str = "easy"

TASK_DEFINITIONS = [
    TaskDefinition(id="easy",   name="Easy Scheduling",   description="3 tasks, 2 machines, generous deadlines.",         max_reward=50.0),
    TaskDefinition(id="medium", name="Medium Scheduling", description="5 tasks, 2 machines, mixed priorities, type matching.", max_reward=110.0),
    TaskDefinition(id="hard",   name="Hard Scheduling",   description="7 tasks, 3 machines, dynamic arrivals, conflicts.",    max_reward=120.0),
]


# ─────────────────────────────────────────────
#  Health / Root (HF Space ping — must return 200)
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "environment": "intelligent-task-scheduling", "version": "1.0.0"}


# ─────────────────────────────────────────────
#  POST /reset
# ─────────────────────────────────────────────

@app.post("/reset", response_model=ResetResponse)
def reset(request: ResetRequest = ResetRequest()):
    global env, current_level

    level = request.level.strip().lower()
    current_level = level

    if level == "easy":
        task_configs, machine_configs = generate_easy()
    elif level == "medium":
        task_configs, machine_configs = generate_medium()
    elif level == "hard":
        task_configs, machine_configs = generate_hard()
    else:
        raise HTTPException(status_code=400, detail=f"Unknown level '{level}'. Use 'easy', 'medium', or 'hard'.")

    env = SchedulerEnv(task_configs, machine_configs)
    obs = env.reset()

    return ResetResponse(observation=obs, wait_action=env.WAIT_ACTION)


# ─────────────────────────────────────────────
#  POST /step
# ─────────────────────────────────────────────

@app.post("/step", response_model=StepResponse)
def step(request: ActionRequest):
    if env is None:
        raise HTTPException(status_code=400, detail="No environment. Call POST /reset first.")
    if env._done:
        raise HTTPException(status_code=400, detail="Episode finished. Call POST /reset to start a new one.")

    obs, reward, done, info = env.step(request.action)

    return StepResponse(observation=obs, reward=float(reward), done=done, info=info)


# ─────────────────────────────────────────────
#  GET /state
# ─────────────────────────────────────────────

@app.get("/state", response_model=StateResponse)
def state():
    if env is None:
        raise HTTPException(status_code=400, detail="No environment. Call POST /reset first.")

    return StateResponse(observation=env.state(), done=env._done, wait_action=env.WAIT_ACTION)


# ─────────────────────────────────────────────
#  GET /tasks — enumerate available tasks
# ─────────────────────────────────────────────

@app.get("/tasks", response_model=List[TaskDefinition])
def list_tasks():
    return TASK_DEFINITIONS


# ─────────────────────────────────────────────
#  POST /grade — grade a completed episode
# ─────────────────────────────────────────────

@app.post("/grade", response_model=GradeResult)
def grade(invalid_actions: int = 0):
    if env is None:
        raise HTTPException(status_code=400, detail="No environment. Call POST /reset first.")
    if not env._done:
        raise HTTPException(status_code=400, detail="Episode not finished yet. Complete it before grading.")

    grader = Grader(current_level)
    report = grader.evaluate(env, invalid_actions)

    return GradeResult(
        task_id=current_level,
        score=report["Overall"],
        details=report
    )


# ─────────────────────────────────────────────
#  Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
