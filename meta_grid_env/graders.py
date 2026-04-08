from __future__ import annotations

from meta_grid_env.models import LoadBalancerState, TaskDefinition, TaskGrade


def grade_task(task: TaskDefinition, state: LoadBalancerState) -> TaskGrade:
    overload_score = max(
        0.0,
        1.0 - (state.cumulative_overload_events / max(task.grader_targets["max_overloads"], 1.0)),
    )
    unmet_score = max(
        0.0,
        1.0 - (state.cumulative_unmet_demand_mw / (task.max_steps * task.grader_targets["max_unmet_demand"])),
    )
    load_shed_score = max(
        0.0,
        1.0 - (state.cumulative_load_shed_mw / (task.max_steps * task.grader_targets["max_load_shed"])),
    )
    budget_score = 1.0 if state.budget_remaining_lakh_rs >= task.grader_targets["min_budget_remaining"] else max(
        0.0,
        state.budget_remaining_lakh_rs / max(task.grader_targets["min_budget_remaining"], 1.0),
    )
    score = round((0.35 * overload_score) + (0.35 * unmet_score) + (0.15 * load_shed_score) + (0.15 * budget_score), 3)
    passed = score >= 0.7
    summary = (
        f"overloads={state.cumulative_overload_events}, "
        f"unmet={state.cumulative_unmet_demand_mw:.1f}MW, "
        f"load_shed={state.cumulative_load_shed_mw:.1f}MW, "
        f"budget_left={state.budget_remaining_lakh_rs:.1f}L"
    )
    return TaskGrade(
        task_name=task.name,
        difficulty=task.difficulty,
        score=min(max(score, 0.0), 1.0),
        passed=passed,
        summary=summary,
    )
