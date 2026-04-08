from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ActionType(str, Enum):
    REDISTRIBUTE_POWER = "redistribute_power"
    ACTIVATE_BACKUP_GENERATOR = "activate_backup_generator"
    SCHEDULE_LOAD_SHEDDING = "schedule_load_shedding"
    BUY_POWER_FROM_GRID = "buy_power_from_grid"
    NOOP = "noop"


class RegionStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    priority: Literal["critical", "industrial", "residential"]
    demand_mw: float = Field(ge=0)
    allocated_power_mw: float = Field(ge=0)
    unmet_demand_mw: float = Field(ge=0)
    load_shed_mw: float = Field(ge=0)


class TransformerStatus(BaseModel):
    model_config = ConfigDict(extra="forbid")

    region: str
    capacity_mw: float = Field(gt=0)
    current_load_mw: float = Field(ge=0)
    load_ratio: float = Field(ge=0)
    overloaded: bool


class GridReward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stability_score: float
    cost_efficiency_score: float
    overload_penalty: float
    blackout_penalty: float
    progress_bonus: float
    total: float


class LoadBalancerAction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    source_region: Optional[str] = None
    target_region: Optional[str] = None
    region: Optional[str] = None
    megawatts: float = Field(default=0.0, ge=0.0)
    duration_hours: int = Field(default=1, ge=1, le=4)
    rationale: str = ""


class LoadBalancerObservation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    done: bool = False
    reward: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    task_name: str
    benchmark_name: str = "meta_hackathon_scaler"
    step_index: int = Field(ge=0)
    max_steps: int = Field(gt=0)
    temperature_c: float
    power_generation_mw: float = Field(ge=0)
    backup_generation_mw: float = Field(ge=0)
    purchased_power_mw: float = Field(ge=0)
    spot_purchase_price_per_mw: float = Field(ge=0)
    budget_remaining_lakh_rs: float
    total_city_demand_mw: float = Field(ge=0)
    unmet_demand_mw: float = Field(ge=0)
    total_load_shed_mw: float = Field(ge=0)
    overloaded_transformers_count: int = Field(ge=0)
    blackout_regions: List[str] = Field(default_factory=list)
    regions: List[RegionStatus]
    transformers: List[TransformerStatus]
    last_action_summary: str = ""
    grader_hint: str = ""


class LoadBalancerState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    episode_id: str
    task_name: str
    step_count: int = Field(ge=0)
    max_steps: int = Field(gt=0)
    budget_remaining_lakh_rs: float
    cumulative_cost_lakh_rs: float
    cumulative_unmet_demand_mw: float
    cumulative_load_shed_mw: float
    cumulative_overload_events: int
    backup_active: bool
    completed: bool
    success_score: float = Field(ge=0.0, le=1.0)


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: Optional[str] = None
    seed: int = Field(default=0, ge=0)


class ResetResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: LoadBalancerObservation
    reward: Optional[float] = None
    done: bool = False


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: LoadBalancerAction


class StepResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: LoadBalancerObservation
    reward: float
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class StateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: LoadBalancerState


class TaskDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    difficulty: Literal["easy", "medium", "hard"]
    objective: str
    max_steps: int = Field(gt=0)
    region_priorities: Dict[str, Literal["critical", "industrial", "residential"]]
    transformer_capacities_mw: Dict[str, float]
    temperature_profile_c: List[float]
    generation_profile_mw: List[float]
    market_price_profile_lakh_rs_per_mw: List[float]
    market_purchase_limit_mw: List[float]
    base_demands_mw: Dict[str, List[float]]
    backup_generator_capacity_mw: float = Field(ge=0)
    backup_generator_cost_lakh_rs: float = Field(ge=0)
    load_shed_penalty_weight: float = Field(ge=0)
    unmet_demand_penalty_weight: float = Field(ge=0)
    overload_penalty_weight: float = Field(ge=0)
    budget_lakh_rs: float = Field(gt=0)
    grader_targets: Dict[str, float]
    grader_hint: str


class TaskGrade(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_name: str
    difficulty: str
    score: float = Field(ge=0.0, le=1.0)
    passed: bool
    summary: str
