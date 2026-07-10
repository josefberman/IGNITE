FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxrender1 libxext6 libsm6 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY backend ./backend
COPY ml ./ml
COPY models ./models
COPY sample_start_molecules.csv ./
COPY scripts ./scripts

RUN pip install --no-cache-dir -e ".[dev]"

RUN mkdir -p /app/data
EXPOSE 8000
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
