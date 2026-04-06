import os
import sys

# Add the root directory to the python path so we can import from `env`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from env.scheduler_env import SchedulerEnv
from env.scenarios import generate_easy

# Provide default simple configs for our global instance initialization
task_configs, machine_configs = generate_easy()

# 2. Initialize a global instance
env = SchedulerEnv(task_configs, machine_configs)

app = FastAPI(
    title="Intelligent Task Scheduling API",
    description="REST API wrapping the SchedulerEnv environment."
)


class ActionRequest(BaseModel):
    """Input JSON format for the POST /step endpoint."""
    action: int = Field(
        ..., 
        description="The internal action ID (task * num_machines + machine).",
        examples=[0]
    )


class ResetRequest(BaseModel):
    """Input JSON format for the POST /reset endpoint."""
    level: str = Field(
        default="easy",
        description="The difficulty level to load: 'easy', 'medium', or 'hard'."
    )


# 3. Implement endpoints

@app.post("/reset")
def reset(request: ResetRequest):
    """
    Starts a new episode and returns the initial observation.
    
    Example Request:
    {
        "level": "hard"
    }

    Example Response format:
    {
        "current_time": 0,
        "num_pending_tasks": 3,
        ...
    }
    """
    global env, task_configs, machine_configs

    level = request.level.strip().lower()
    
    # Generate the correct difficulty scenario
    from env.scenarios import generate_easy, generate_medium, generate_hard
    if level == "easy":
        task_configs, machine_configs = generate_easy()
    elif level == "medium":
        task_configs, machine_configs = generate_medium()
    elif level == "hard":
        task_configs, machine_configs = generate_hard()
    else:
        raise HTTPException(
            status_code=400, 
            detail=f"Unknown level '{level}'. Choose 'easy', 'medium', or 'hard'."
        )

    # Re-initialize the environment
    env = SchedulerEnv(task_configs, machine_configs)
    
    obs = env.reset()
    
    # Include the wait_action so the client knows what integer to send to WAIT
    obs["wait_action"] = env.WAIT_ACTION
    return obs


@app.post("/step")
def step(request: ActionRequest):
    """
    Executes a single step given the action.
    
    Example Request:
    {
        "action": 0
    }
    
    Example Response:
    {
        "observation": {
            "current_time": 0,
            ...
        },
        "reward": 0.0,
        "done": false,
        "info": {
            "assigned": "Task 1 → Machine 1",
            "step": 1,
            "total_reward": 0.0,
            ...
        }
    }
    """
    # If the episode is already done, let the user know cleanly.
    if env._done:
        raise HTTPException(
            status_code=400, 
            detail="Episode is already finished. Call POST /reset to start a new one."
        )

    # Validates and executes action
    obs, reward, done, info = env.step(request.action)

    return {
        "observation": obs,
        "reward": float(reward),
        "done": done,
        "info": info
    }


@app.get("/state")
def state():
    """
    Gets the current observation (state) of the environment without advancing time.
    
    Example Response format:
    {
        "current_time": 2,
        "num_pending_tasks": 1,
        "num_active_tasks": 1,
        "num_completed_tasks": 1,
        "num_missed_tasks": 0,
        "tasks": [...],
        "machines": [...]
    }
    """
    return env.state()


if __name__ == "__main__":
    import uvicorn
    # To run this server, use: uvicorn api.server:app --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)
