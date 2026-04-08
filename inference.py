from __future__ import annotations

import asyncio
import json
import os
import textwrap
from typing import List, Optional

from openai import OpenAI

from meta_grid_env import LoadBalancerAction, SmartGridEnv
from meta_grid_env.client import action_to_log_string
from meta_grid_env.graders import grade_task
from meta_grid_env.models import ActionType, LoadBalancerObservation
from meta_grid_env.tasks import get_task, list_tasks


IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
ENV_BASE_URL = os.getenv("ENV_BASE_URL")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "meta_hackathon_scaler"
SUCCESS_SCORE_THRESHOLD = 0.70

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are operating a smart electricity dispatcher for an Indian city.
    Return exactly one compact JSON object with fields:
    action_type, source_region, target_region, region, megawatts, duration_hours, rationale.
    Choose only one of these action_type values:
    redistribute_power, activate_backup_generator, schedule_load_shedding, buy_power_from_grid, noop.
    Prefer protecting critical regions, preventing transformer overloads, and avoiding blackouts.
    """
).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def build_user_prompt(task_name: str, observation: LoadBalancerObservation) -> str:
    region_lines = [
        f"{region.name}: demand={region.demand_mw:.1f}, unmet={region.unmet_demand_mw:.1f}, shed={region.load_shed_mw:.1f}"
        for region in observation.regions
    ]
    transformer_lines = [
        f"{transformer.region}: ratio={transformer.load_ratio:.2f}, overloaded={str(transformer.overloaded).lower()}"
        for transformer in observation.transformers
    ]
    return textwrap.dedent(
        f"""
        Task: {task_name}
        Objective: {get_task(task_name).objective}
        Step: {observation.step_index}/{observation.max_steps}
        Temperature: {observation.temperature_c:.1f}C
        Base generation: {observation.power_generation_mw:.1f} MW
        Purchased power available price: {observation.spot_purchase_price_per_mw:.2f} lakh/MW
        Budget remaining: {observation.budget_remaining_lakh_rs:.1f} lakh
        Total demand: {observation.total_city_demand_mw:.1f} MW
        Unmet demand: {observation.unmet_demand_mw:.1f} MW
        Overloaded transformers: {observation.overloaded_transformers_count}
        Blackout regions: {",".join(observation.blackout_regions) if observation.blackout_regions else "none"}
        Regions:
        {"; ".join(region_lines)}
        Transformers:
        {"; ".join(transformer_lines)}
        Hint:
        {observation.grader_hint}
        Return the best next action as JSON.
        """
    ).strip()


def heuristic_action(task_name: str, observation: LoadBalancerObservation) -> LoadBalancerAction:
    overloaded = [item for item in observation.transformers if item.overloaded]
    regions_by_unmet = sorted(observation.regions, key=lambda item: item.unmet_demand_mw, reverse=True)
    critical_region = next((item.name for item in observation.regions if item.priority == "critical"), None)

    if observation.step_index == 0 and observation.temperature_c >= 40.0:
        return LoadBalancerAction(action_type=ActionType.ACTIVATE_BACKUP_GENERATOR, rationale="Heatwave reserve")
    if overloaded:
        source = max(observation.regions, key=lambda item: max(item.allocated_power_mw - item.demand_mw, 0.0)).name
        return LoadBalancerAction(
            action_type=ActionType.REDISTRIBUTE_POWER,
            source_region=source,
            target_region=overloaded[0].region,
            megawatts=12.0,
            rationale="Cool transformer stress",
        )
    if observation.unmet_demand_mw > 20.0 and observation.budget_remaining_lakh_rs > 12.0:
        return LoadBalancerAction(
            action_type=ActionType.BUY_POWER_FROM_GRID,
            megawatts=min(18.0, observation.unmet_demand_mw),
            rationale="Cover supply gap",
        )
    if regions_by_unmet and regions_by_unmet[0].unmet_demand_mw > 10.0 and critical_region and regions_by_unmet[0].name != critical_region:
        return LoadBalancerAction(
            action_type=ActionType.SCHEDULE_LOAD_SHEDDING,
            region=regions_by_unmet[0].name,
            megawatts=min(10.0, regions_by_unmet[0].unmet_demand_mw),
            rationale="Selective relief",
        )
    return LoadBalancerAction(action_type=ActionType.NOOP, rationale=f"Stable dispatch for {task_name}")


def model_action(client: OpenAI, task_name: str, observation: LoadBalancerObservation) -> LoadBalancerAction:
    if not API_KEY:
        return heuristic_action(task_name, observation)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0.0,
            max_tokens=180,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(task_name, observation)},
            ],
        )
        content = (completion.choices[0].message.content or "").strip()
        payload = json.loads(content)
        return LoadBalancerAction.model_validate(payload)
    except Exception:
        return heuristic_action(task_name, observation)


async def run_task(client: OpenAI, task_name: str) -> tuple[float, List[float]]:
    env = await SmartGridEnv.from_url(ENV_BASE_URL) if ENV_BASE_URL else await SmartGridEnv.from_docker_image(IMAGE_NAME)
    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0
    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = await env.reset(task_name=task_name)
        for step in range(1, get_task(task_name).max_steps + 1):
            observation = result.observation
            if result.done:
                break
            action = model_action(client, task_name, observation)
            result = await env.step(action)
            reward = result.reward
            steps_taken = step
            rewards.append(reward)
            error = result.info.get("last_action_error") or None
            log_step(step=step, action=action_to_log_string(action), reward=reward, done=result.done, error=error)
            if result.done:
                break

        final_state = await env.state()
        score = grade_task(get_task(task_name), final_state).score
        success = score >= SUCCESS_SCORE_THRESHOLD
    finally:
        await env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score, rewards


async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY or "missing-key")
    task_filter = os.getenv("SMART_GRID_TASK")
    tasks = [task_filter] if task_filter else list_tasks()
    for task_name in tasks:
        await run_task(client, task_name)


if __name__ == "__main__":
    asyncio.run(main())
