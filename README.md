# MLOps CI/CD Pipeline

An Iris logistic regression classifier and a GitHub Actions workflow that validates the model before simulated deployment. The classifier uses a stratified 80/20 split with seed 42. StandardScaler is fitted on the training split as part of the model pipeline.

## Run locally

From the repository root, use Python 3.11 and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m compileall -q app.py src tests
.\.venv\Scripts\python.exe app.py
.\.venv\Scripts\python.exe -m pytest -v
```

The same validation runs on GitHub for pushes and pull requests targeting `main`. The test job installs dependencies, checks syntax, executes pytest, and runs the application. The deploy job needs a successful test job and a push to `main`. Its two commands print deployment simulation messages.

## Demonstrate the gate

The test requires 0.80 accuracy by default. To demonstrate a failure remotely in this dedicated lab repository, temporarily change the default to 0.99 in `tests/test_model.py`, commit and push. The test should fail and deployment should be skipped. Restore 0.80 with a separate commit and push again. Wait for each Actions run to finish before pushing the next scenario.

The same threshold can also be overridden locally without editing the test:

```powershell
$env:MIN_MODEL_ACCURACY = '0.99'
python -m pytest -v
Remove-Item Env:MIN_MODEL_ACCURACY
python -m pytest -v
```
