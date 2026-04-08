from __future__ import annotations

from fastapi import FastAPI

from meta_grid_env.models import LoadBalancerObservation, ResetRequest, StateResponse, StepRequest
from meta_grid_env.server.grid_environment import SmartGridEnvironment


app = FastAPI(title="Meta Hackathon Scaler", version="0.1.0")
ENV = SmartGridEnvironment()


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.get("/schema")
async def schema() -> dict[str, object]:
    return {
        "action": StepRequest.model_json_schema(),
        "observation": LoadBalancerObservation.model_json_schema(),
        "state": StateResponse.model_json_schema(),
    }


@app.post("/reset")
async def reset(request: ResetRequest) -> dict[str, object]:
    response = ENV.reset(task_name=request.task_name)
    return response.model_dump()


@app.post("/step")
async def step(request: StepRequest) -> dict[str, object]:
    response = ENV.step(request.action)
    return response.model_dump()


@app.get("/state")
async def state() -> dict[str, object]:
    return StateResponse(state=ENV.state()).model_dump()


@app.get("/tasks")
async def tasks() -> dict[str, list[str]]:
    return {"tasks": ENV.available_tasks()}
