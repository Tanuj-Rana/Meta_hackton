from __future__ import annotations

import uuid
from copy import deepcopy
from typing import Dict, List

from meta_grid_env.graders import grade_task
from meta_grid_env.models import (
    ActionType,
    GridReward,
    LoadBalancerAction,
    LoadBalancerObservation,
    LoadBalancerState,
    RegionStatus,
    ResetResponse,
    StepResponse,
    TaskDefinition,
    TransformerStatus,
)
from meta_grid_env.tasks import get_task, list_tasks


class SmartGridEnvironment:
    def __init__(self) -> None:
        self.task: TaskDefinition | None = None
        self.episode_id = ""
        self.step_index = 0
        self.budget_remaining = 0.0
        self.cumulative_cost = 0.0
        self.cumulative_unmet = 0.0
        self.cumulative_load_shed = 0.0
        self.cumulative_overloads = 0
        self.backup_active = False
        self.completed = False
        self.last_action_summary = ""
        self.last_reward = 0.0
        self.current_observation: LoadBalancerObservation | None = None

    def available_tasks(self) -> List[str]:
        return list_tasks()

    def reset(self, task_name: str | None = None) -> ResetResponse:
        selected_task = get_task(task_name or list_tasks()[0])
        self.task = selected_task
        self.episode_id = str(uuid.uuid4())
        self.step_index = 0
        self.budget_remaining = selected_task.budget_lakh_rs
        self.cumulative_cost = 0.0
        self.cumulative_unmet = 0.0
        self.cumulative_load_shed = 0.0
        self.cumulative_overloads = 0
        self.backup_active = False
        self.completed = False
        self.last_action_summary = "Environment reset."
        self.last_reward = 0.0
        self.current_observation = self._build_observation(action_summary=self.last_action_summary, reward=0.0, done=False)
        return ResetResponse(observation=self.current_observation, reward=0.0, done=False)

    def step(self, action: LoadBalancerAction) -> StepResponse:
        if self.task is None:
            raise RuntimeError("Environment must be reset before stepping.")
        if self.completed:
            return StepResponse(
                observation=self.current_observation or self._build_observation("Episode already done.", 0.0, True),
                reward=0.0,
                done=True,
                info={"last_action_error": "episode already completed"},
            )

        demand_by_region = self._current_demands()
        generation = self.task.generation_profile_mw[self.step_index]
        market_limit = self.task.market_purchase_limit_mw[self.step_index]
        market_price = self.task.market_price_profile_lakh_rs_per_mw[self.step_index]

        load_shed = {name: 0.0 for name in demand_by_region}
        purchased_power = 0.0
        transfer_adjustment = {name: 0.0 for name in demand_by_region}
        info: Dict[str, str | float] = {"last_action_error": ""}
        action_summary = self._describe_action(action)

        if action.action_type == ActionType.ACTIVATE_BACKUP_GENERATOR:
            if not self.backup_active:
                self.backup_active = True
                self.budget_remaining -= self.task.backup_generator_cost_lakh_rs
                self.cumulative_cost += self.task.backup_generator_cost_lakh_rs
                action_summary = f"Backup generator activated (+{self.task.backup_generator_capacity_mw:.1f} MW reserve)."
            else:
                action_summary = "Backup generator was already active."
        elif action.action_type == ActionType.BUY_POWER_FROM_GRID:
            purchased_power = min(action.megawatts, market_limit)
            purchase_cost = purchased_power * market_price
            if purchase_cost > self.budget_remaining:
                affordable = self.budget_remaining / market_price if market_price > 0 else 0.0
                purchased_power = min(affordable, market_limit)
                purchase_cost = purchased_power * market_price
                info["last_action_error"] = "purchase truncated by remaining budget"
            self.budget_remaining -= purchase_cost
            self.cumulative_cost += purchase_cost
            action_summary = f"Bought {purchased_power:.1f} MW from external grid."
        elif action.action_type == ActionType.SCHEDULE_LOAD_SHEDDING:
            if action.region and action.region in demand_by_region:
                load_shed[action.region] = min(action.megawatts, demand_by_region[action.region] * 0.35)
                action_summary = f"Scheduled {load_shed[action.region]:.1f} MW shedding in {action.region}."
            else:
                info["last_action_error"] = "invalid shedding region"
        elif action.action_type == ActionType.REDISTRIBUTE_POWER:
            if (
                action.source_region
                and action.target_region
                and action.source_region in demand_by_region
                and action.target_region in demand_by_region
                and action.source_region != action.target_region
            ):
                transfer = min(action.megawatts, 18.0)
                transfer_adjustment[action.source_region] -= transfer
                transfer_adjustment[action.target_region] += transfer
                action_summary = f"Shifted {transfer:.1f} MW from {action.source_region} to {action.target_region}."
            else:
                info["last_action_error"] = "invalid redistribution request"
        else:
            action_summary = "Held dispatch steady for one step."

        effective_demand = {
            name: max(0.0, demand - load_shed[name]) for name, demand in demand_by_region.items()
        }
        total_effective_demand = sum(effective_demand.values())
        total_supply = generation + purchased_power + (self.task.backup_generator_capacity_mw if self.backup_active else 0.0)

        allocations = self._allocate_supply(effective_demand, total_supply, transfer_adjustment)
        transformers = []
        regions = []
        blackout_regions: list[str] = []
        unmet_total = 0.0
        overloads = 0

        for region_name, demand in effective_demand.items():
            capacity = self.task.transformer_capacities_mw[region_name]
            allocated = allocations.get(region_name, 0.0)
            current_load = min(allocated, capacity * 1.12)
            load_ratio = current_load / capacity if capacity > 0 else 0.0
            overloaded = load_ratio > 1.0
            if overloaded:
                overloads += 1
            unmet = max(0.0, demand - allocated)
            unmet_total += unmet
            if unmet > 12.0:
                blackout_regions.append(region_name)
            transformers.append(
                TransformerStatus(
                    region=region_name,
                    capacity_mw=capacity,
                    current_load_mw=round(current_load, 2),
                    load_ratio=round(load_ratio, 3),
                    overloaded=overloaded,
                )
            )
            regions.append(
                RegionStatus(
                    name=region_name,
                    priority=self.task.region_priorities[region_name],
                    demand_mw=round(demand_by_region[region_name], 2),
                    allocated_power_mw=round(allocated, 2),
                    unmet_demand_mw=round(unmet, 2),
                    load_shed_mw=round(load_shed[region_name], 2),
                )
            )

        self.cumulative_unmet += unmet_total
        self.cumulative_load_shed += sum(load_shed.values())
        self.cumulative_overloads += overloads

        reward_breakdown = self._compute_reward(
            effective_demand_total=total_effective_demand,
            unmet_total=unmet_total,
            overloads=overloads,
            load_shed_total=sum(load_shed.values()),
            purchased_power=purchased_power,
            blackout_regions=len(blackout_regions),
        )

        self.step_index += 1
        done = self.step_index >= self.task.max_steps
        self.completed = done
        self.last_reward = reward_breakdown.total
        self.last_action_summary = action_summary

        observation = LoadBalancerObservation(
            done=done,
            reward=reward_breakdown.total,
            task_name=self.task.name,
            step_index=self.step_index,
            max_steps=self.task.max_steps,
            temperature_c=self.task.temperature_profile_c[self.step_index - 1],
            power_generation_mw=generation,
            backup_generation_mw=self.task.backup_generator_capacity_mw if self.backup_active else 0.0,
            purchased_power_mw=round(purchased_power, 2),
            spot_purchase_price_per_mw=market_price,
            budget_remaining_lakh_rs=round(self.budget_remaining, 2),
            total_city_demand_mw=round(sum(demand_by_region.values()), 2),
            unmet_demand_mw=round(unmet_total, 2),
            total_load_shed_mw=round(sum(load_shed.values()), 2),
            overloaded_transformers_count=overloads,
            blackout_regions=blackout_regions,
            regions=regions,
            transformers=transformers,
            last_action_summary=action_summary,
            grader_hint=self.task.grader_hint,
            metadata={"reward_breakdown": reward_breakdown.model_dump(), "last_action_error": info["last_action_error"] or None},
        )
        self.current_observation = observation
        info["grade"] = grade_task(self.task, self.state()).model_dump()
        return StepResponse(observation=observation, reward=reward_breakdown.total, done=done, info=deepcopy(info))

    def state(self) -> LoadBalancerState:
        if self.task is None:
            raise RuntimeError("Environment has not been reset.")
        grade = grade_task(self.task, self._state_without_grade())
        return LoadBalancerState(
            episode_id=self.episode_id,
            task_name=self.task.name,
            step_count=self.step_index,
            max_steps=self.task.max_steps,
            budget_remaining_lakh_rs=round(self.budget_remaining, 2),
            cumulative_cost_lakh_rs=round(self.cumulative_cost, 2),
            cumulative_unmet_demand_mw=round(self.cumulative_unmet, 2),
            cumulative_load_shed_mw=round(self.cumulative_load_shed, 2),
            cumulative_overload_events=self.cumulative_overloads,
            backup_active=self.backup_active,
            completed=self.completed,
            success_score=grade.score,
        )

    def _state_without_grade(self) -> LoadBalancerState:
        if self.task is None:
            raise RuntimeError("Environment has not been reset.")
        return LoadBalancerState(
            episode_id=self.episode_id,
            task_name=self.task.name,
            step_count=self.step_index,
            max_steps=self.task.max_steps,
            budget_remaining_lakh_rs=round(self.budget_remaining, 2),
            cumulative_cost_lakh_rs=round(self.cumulative_cost, 2),
            cumulative_unmet_demand_mw=round(self.cumulative_unmet, 2),
            cumulative_load_shed_mw=round(self.cumulative_load_shed, 2),
            cumulative_overload_events=self.cumulative_overloads,
            backup_active=self.backup_active,
            completed=self.completed,
            success_score=0.0,
        )

    def _build_observation(self, action_summary: str, reward: float, done: bool) -> LoadBalancerObservation:
        if self.task is None:
            raise RuntimeError("Environment has not been reset.")
        demand_by_region = self._current_demands()
        regions = [
            RegionStatus(
                name=region_name,
                priority=self.task.region_priorities[region_name],
                demand_mw=round(demand_value, 2),
                allocated_power_mw=0.0,
                unmet_demand_mw=round(demand_value, 2),
                load_shed_mw=0.0,
            )
            for region_name, demand_value in demand_by_region.items()
        ]
        transformers = [
            TransformerStatus(
                region=region_name,
                capacity_mw=capacity,
                current_load_mw=0.0,
                load_ratio=0.0,
                overloaded=False,
            )
            for region_name, capacity in self.task.transformer_capacities_mw.items()
        ]
        return LoadBalancerObservation(
            done=done,
            reward=reward,
            task_name=self.task.name,
            step_index=self.step_index,
            max_steps=self.task.max_steps,
            temperature_c=self.task.temperature_profile_c[self.step_index],
            power_generation_mw=self.task.generation_profile_mw[self.step_index],
            backup_generation_mw=0.0,
            purchased_power_mw=0.0,
            spot_purchase_price_per_mw=self.task.market_price_profile_lakh_rs_per_mw[self.step_index],
            budget_remaining_lakh_rs=self.budget_remaining,
            total_city_demand_mw=round(sum(demand_by_region.values()), 2),
            unmet_demand_mw=round(sum(demand_by_region.values()), 2),
            total_load_shed_mw=0.0,
            overloaded_transformers_count=0,
            blackout_regions=[],
            regions=regions,
            transformers=transformers,
            last_action_summary=action_summary,
            grader_hint=self.task.grader_hint,
            metadata={"available_tasks": self.available_tasks()},
        )

    def _current_demands(self) -> Dict[str, float]:
        if self.task is None:
            raise RuntimeError("Environment has not been reset.")
        return {region_name: profile[self.step_index] for region_name, profile in self.task.base_demands_mw.items()}

    def _allocate_supply(
        self,
        effective_demand: Dict[str, float],
        total_supply: float,
        transfer_adjustment: Dict[str, float],
    ) -> Dict[str, float]:
        if self.task is None:
            raise RuntimeError("Environment has not been reset.")
        weighted_total = 0.0
        weights: Dict[str, float] = {}
        priority_weight = {"critical": 1.25, "industrial": 1.05, "residential": 1.0}
        for region_name, demand in effective_demand.items():
            weight = priority_weight[self.task.region_priorities[region_name]]
            weights[region_name] = weight
            weighted_total += demand * weight

        allocations: Dict[str, float] = {}
        for region_name, demand in effective_demand.items():
            base = (total_supply * demand * weights[region_name] / weighted_total) if weighted_total > 0 else 0.0
            allocations[region_name] = max(0.0, base + transfer_adjustment[region_name])
        return allocations

    def _compute_reward(
        self,
        effective_demand_total: float,
        unmet_total: float,
        overloads: int,
        load_shed_total: float,
        purchased_power: float,
        blackout_regions: int,
    ) -> GridReward:
        if self.task is None:
            raise RuntimeError("Environment has not been reset.")
        served_ratio = 1.0 - (unmet_total / effective_demand_total if effective_demand_total > 0 else 0.0)
        stability_score = 0.55 * served_ratio
        cost_efficiency_score = max(0.0, 0.2 - (0.004 * purchased_power) - (0.015 if self.backup_active else 0.0))
        overload_penalty = self.task.overload_penalty_weight * overloads * 0.18
        blackout_penalty = (blackout_regions * 0.22) + (self.task.load_shed_penalty_weight * load_shed_total * 0.01)
        progress_bonus = 0.12 if unmet_total <= self.task.grader_targets["max_unmet_demand"] else 0.03
        total = stability_score + cost_efficiency_score + progress_bonus - overload_penalty - blackout_penalty
        return GridReward(
            stability_score=round(stability_score, 3),
            cost_efficiency_score=round(cost_efficiency_score, 3),
            overload_penalty=round(overload_penalty, 3),
            blackout_penalty=round(blackout_penalty, 3),
            progress_bonus=round(progress_bonus, 3),
            total=round(min(max(total, 0.0), 1.0), 3),
        )

    def _describe_action(self, action: LoadBalancerAction) -> str:
        if action.action_type == ActionType.REDISTRIBUTE_POWER:
            return f"Redistribute {action.megawatts:.1f} MW from {action.source_region} to {action.target_region}."
        if action.action_type == ActionType.ACTIVATE_BACKUP_GENERATOR:
            return "Activate backup generator."
        if action.action_type == ActionType.SCHEDULE_LOAD_SHEDDING:
            return f"Shed {action.megawatts:.1f} MW in {action.region}."
        if action.action_type == ActionType.BUY_POWER_FROM_GRID:
            return f"Buy {action.megawatts:.1f} MW from market."
        return "No operation."
