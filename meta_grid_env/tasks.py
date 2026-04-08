from __future__ import annotations

from meta_grid_env.models import TaskDefinition


TASKS = {
    "summer_peak_relief": TaskDefinition(
        name="summer_peak_relief",
        difficulty="easy",
        objective="Keep hospital and residential demand stable during a summer AC spike while avoiding transformer overloads.",
        max_steps=6,
        region_priorities={
            "north": "residential",
            "central": "critical",
            "south": "residential",
        },
        transformer_capacities_mw={"north": 125.0, "central": 140.0, "south": 120.0},
        temperature_profile_c=[39.0, 41.0, 43.0, 42.0, 40.0, 39.0],
        generation_profile_mw=[290.0, 285.0, 280.0, 278.0, 282.0, 288.0],
        market_price_profile_lakh_rs_per_mw=[0.13, 0.16, 0.18, 0.17, 0.15, 0.14],
        market_purchase_limit_mw=[45.0, 50.0, 60.0, 60.0, 45.0, 40.0],
        base_demands_mw={
            "north": [112.0, 120.0, 130.0, 126.0, 116.0, 110.0],
            "central": [88.0, 92.0, 95.0, 94.0, 91.0, 89.0],
            "south": [96.0, 101.0, 112.0, 108.0, 101.0, 97.0],
        },
        backup_generator_capacity_mw=28.0,
        backup_generator_cost_lakh_rs=3.5,
        load_shed_penalty_weight=0.45,
        unmet_demand_penalty_weight=0.85,
        overload_penalty_weight=0.9,
        budget_lakh_rs=48.0,
        grader_targets={
            "max_overloads": 1.0,
            "max_unmet_demand": 40.0,
            "max_load_shed": 18.0,
            "min_budget_remaining": 8.0,
        },
        grader_hint="Protect the central critical corridor and keep total unmet demand very low.",
    ),
    "monsoon_outage_recovery": TaskDefinition(
        name="monsoon_outage_recovery",
        difficulty="medium",
        objective="Handle a rainy-day generation dip and restore industrial service with minimal purchased power.",
        max_steps=7,
        region_priorities={
            "north": "industrial",
            "central": "critical",
            "south": "residential",
        },
        transformer_capacities_mw={"north": 150.0, "central": 135.0, "south": 115.0},
        temperature_profile_c=[31.0, 30.0, 29.0, 28.0, 29.0, 30.0, 31.0],
        generation_profile_mw=[300.0, 260.0, 235.0, 240.0, 255.0, 268.0, 275.0],
        market_price_profile_lakh_rs_per_mw=[0.12, 0.14, 0.21, 0.22, 0.19, 0.16, 0.13],
        market_purchase_limit_mw=[35.0, 35.0, 50.0, 50.0, 45.0, 35.0, 30.0],
        base_demands_mw={
            "north": [118.0, 128.0, 135.0, 138.0, 130.0, 125.0, 120.0],
            "central": [92.0, 94.0, 96.0, 100.0, 98.0, 95.0, 93.0],
            "south": [86.0, 88.0, 90.0, 95.0, 96.0, 92.0, 88.0],
        },
        backup_generator_capacity_mw=32.0,
        backup_generator_cost_lakh_rs=4.0,
        load_shed_penalty_weight=0.55,
        unmet_demand_penalty_weight=0.95,
        overload_penalty_weight=1.1,
        budget_lakh_rs=55.0,
        grader_targets={
            "max_overloads": 1.0,
            "max_unmet_demand": 52.0,
            "max_load_shed": 22.0,
            "min_budget_remaining": 10.0,
        },
        grader_hint="Use redistribution before buying costly market power; keep industrial north online where possible.",
    ),
    "festival_budget_crunch": TaskDefinition(
        name="festival_budget_crunch",
        difficulty="hard",
        objective="Balance festival lighting demand, hospital reliability, and a tight purchase budget across the city.",
        max_steps=8,
        region_priorities={
            "north": "industrial",
            "central": "critical",
            "south": "residential",
            "west": "residential",
        },
        transformer_capacities_mw={"north": 132.0, "central": 145.0, "south": 118.0, "west": 110.0},
        temperature_profile_c=[34.0, 35.0, 36.0, 37.0, 36.0, 35.0, 34.0, 33.0],
        generation_profile_mw=[320.0, 312.0, 300.0, 292.0, 290.0, 295.0, 305.0, 312.0],
        market_price_profile_lakh_rs_per_mw=[0.15, 0.17, 0.24, 0.28, 0.27, 0.22, 0.18, 0.16],
        market_purchase_limit_mw=[30.0, 35.0, 45.0, 45.0, 40.0, 35.0, 30.0, 25.0],
        base_demands_mw={
            "north": [102.0, 108.0, 116.0, 118.0, 114.0, 110.0, 105.0, 102.0],
            "central": [98.0, 101.0, 104.0, 108.0, 110.0, 107.0, 102.0, 99.0],
            "south": [92.0, 99.0, 108.0, 118.0, 120.0, 115.0, 105.0, 96.0],
            "west": [84.0, 88.0, 96.0, 110.0, 114.0, 108.0, 95.0, 88.0],
        },
        backup_generator_capacity_mw=26.0,
        backup_generator_cost_lakh_rs=4.8,
        load_shed_penalty_weight=0.7,
        unmet_demand_penalty_weight=1.1,
        overload_penalty_weight=1.25,
        budget_lakh_rs=52.0,
        grader_targets={
            "max_overloads": 1.0,
            "max_unmet_demand": 65.0,
            "max_load_shed": 24.0,
            "min_budget_remaining": 6.0,
        },
        grader_hint="The central corridor must stay stable; budget discipline matters more than in the easier tasks.",
    ),
}


def list_tasks() -> list[str]:
    return list(TASKS.keys())


def get_task(task_name: str) -> TaskDefinition:
    if task_name not in TASKS:
        raise KeyError(f"Unknown task: {task_name}")
    return TASKS[task_name]
