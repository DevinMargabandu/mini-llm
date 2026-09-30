FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements-serve.txt ./
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements-serve.txt

COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY checkpoints ./checkpoints

EXPOSE 8000
ENV MINILLM_CHECKPOINT_DIR=/app/checkpoints \
    MINILLM_MAX_BATCH_SIZE=16

CMD ["uvicorn", "minillm.serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
