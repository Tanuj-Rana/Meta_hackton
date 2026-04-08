FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md /app/
COPY meta_grid_env /app/meta_grid_env
COPY server /app/server
COPY inference.py /app/inference.py
COPY openenv.yaml /app/openenv.yaml

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "meta_grid_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
