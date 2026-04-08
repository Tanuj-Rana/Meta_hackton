# Meta Hackathon Scaler

`meta_hackathon_scaler` is a realistic smart-electricity dispatch simulator inspired by Indian city grid operations handled by DISCOMs and infrastructure companies like BHEL. The agent acts like a control-room operator who must keep demand and supply balanced during heat waves, outages, and festival peaks.

## Why this is a strong hackathon project

- It simulates a real city power-management task instead of a toy game.
- The environment exposes typed action, observation, reward, and state models.
- It contains three deterministic tasks with graders: easy, medium, and hard.
- Rewards provide partial progress signals at every step.
- It includes a root `inference.py`, root `Dockerfile`, and `openenv.yaml`.

## Environment description

At every step the agent observes:

- `region_demand`
- `transformer_load`
- `temperature`
- `power_generation`
- `spot_purchase_price`
- `budget_remaining`

The agent can take one of these actions:

- `redistribute_power`
- `activate_backup_generator`
- `schedule_load_shedding`
- `buy_power_from_grid`
- `noop`

The simulator tracks:

- unmet demand and blackout-prone regions
- transformer overloads
- purchased power cost
- load shedding volume
- backup generator usage

## Reward design

Reward is in `[0.0, 1.0]` at every step and includes:

- positive signal for serving demand and keeping the grid stable
- penalty for transformer overloads
- penalty for blackouts and heavy load shedding
- penalty for expensive external power purchases
- progress bonus when the step stays under task thresholds

## Tasks

### 1. `summer_peak_relief` (easy)

Manage a summer AC surge while protecting the central critical corridor and preventing overloads.

### 2. `monsoon_outage_recovery` (medium)

Recover from a generation dip during monsoon conditions while keeping industrial output stable and external purchases under control.

### 3. `festival_budget_crunch` (hard)

Handle festival lighting demand spikes under a tighter budget, while keeping hospital/critical service reliability high.

## Graders

Each task has a deterministic grader that returns a normalized score in `[0.0, 1.0]` based on:

- cumulative overload events
- cumulative unmet demand
- cumulative load shed
- budget remaining

A task is treated as passed when the final score is `>= 0.70`.

## Project structure

```text
meta_grid_env/
  __init__.py
  client.py
  graders.py
  models.py
  tasks.py
  server/
    app.py
    grid_environment.py
inference.py
openenv.yaml
Dockerfile
README.md
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run locally

Start the environment server:

```bash
uvicorn meta_grid_env.server.app:app --host 0.0.0.0 --port 8000
```

Or build and run with Docker:

```bash
docker build -t meta-grid-env .
docker run -p 8000:8000 meta-grid-env
```

## Inference

The hackathon requires:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`
- optionally `LOCAL_IMAGE_NAME`

Run:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your_token"
export LOCAL_IMAGE_NAME="meta-grid-env"
python3 inference.py
```

The script:

- uses the OpenAI client for model calls
- prints `[START]`, `[STEP]`, and `[END]` lines in the required format
- evaluates all three tasks by default

## Baseline scores

Deterministic heuristic-fallback scores from a local smoke run:

- `summer_peak_relief`: `0.785`
- `monsoon_outage_recovery`: `0.717`
- `festival_budget_crunch`: `0.650`

An instruct model with the OpenAI client can improve on these by making better step-by-step tradeoffs.

## Deployment to Hugging Face Spaces

1. Create a Docker Space.
2. Add the repo contents.
3. Ensure the Space exposes port `8000`.
4. Add your environment variables in the Space settings.
5. Tag the Space with `openenv`.

The `/reset` endpoint is available at:

```text
https://<your-space>.hf.space/reset
```

## Validation checklist

Before submitting:

```bash
docker build .
openenv validate
python3 inference.py
```

Then run the hackathon validation script against your HF Space URL.
