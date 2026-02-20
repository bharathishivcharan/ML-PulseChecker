# ML-PulseChecker

This project is a **churn prediction ML pipeline**:

- Data preprocessing (`build_user_table.py`, `generate_events.py`, `label_churn.py`)
- Train RandomForest model (`train_churn_model.py`)
- Deploy ML model via FastAPI (`deploy_api.py`)

## How to Run

1. Clone the repo
2. Create a virtual environment
3. Install dependencies:

Run `pip install -r requirements.txt`

4. Train the model:

Run `python src/train_churn_model.py`

5. Run API:

Run `uvicorn src.deploy_api:app --reload`

6. Test `/predict` endpoint via browser or `curl`
