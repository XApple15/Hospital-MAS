# Hospital-MAS

# Start frontend and fastapi server 
uv run uvicorn hospitalmas.server:app --port 8000 --host 127.0.0.1

# run tests
uv run pytest tests

# run tests from .csv
python3 -m hospitalmas.eval_runner --prognosis "Hepatitis E" --variant 100 --limit 1 --csv ./PatientCases.csv

