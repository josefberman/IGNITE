# IGNITE

Computational search for energetic-molecule candidates. The system and the results are in [docs/IGNITE_System_Thesis.md](docs/IGNITE_System_Thesis.md).

Outputs are computational candidates only, not a synthesis instruction or a hazard assessment.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,sascore,torch]"
uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8080
```

```bash
cd frontend && npm install && npm run dev
```

The GUI is at http://localhost:5173 and proxies `/api` to port 8080.

## Train

```bash
python scripts/train_arch1_parallel.py
python scripts/train_arch2_cascade_fc.py
python scripts/train_arch3_cascade.py
```

The default pairs file is `data/processed/unified_pairs.csv`. The split seed is 42.

## License

Apache License 2.0. See [LICENCE.md](LICENCE.md) and [docs/LICENSING.md](docs/LICENSING.md).
