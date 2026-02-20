import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import uuid
import random
import os

np.random.seed(42)
random.seed(42)

NUM_USERS = 5000
DAYS = 60

EVENT_TYPES = [
    "app_open",
    "view_item",
    "add_to_cart",
    "purchase",
    "search",
    "feature_use"
]

DEVICES = ["iOS", "Android", "Web"]
EXPERIMENT_VARIANTS = ["control", "variant_a"]

def generate_user_ids(n):
    return [str(uuid.uuid4()) for _ in range(n)]

def generate_events(num_users=NUM_USERS, days=DAYS):
    users = generate_user_ids(num_users)
    start_date = datetime.today() - timedelta(days=days)

    events = []
    for day in range(days):
        current_date = start_date + timedelta(days=day)

        # realistic DAU fluctuation: 40% - 70% of users active each day
        daily_active_users = np.random.choice(
            users,
            size=int(num_users * np.random.uniform(0.4, 0.7)),
            replace=False
        )

        for user in daily_active_users:
            sessions = np.random.randint(1, 4)  # 1..3 sessions/day
            for _ in range(sessions):
                session_id = str(uuid.uuid4())
                num_events = np.random.randint(1, 10)
                for _ in range(num_events):
                    event_type = random.choice(EVENT_TYPES)
                    revenue = 0.0
                    if event_type == "purchase":
                        # exponential to simulate many small purchases + few large
                        revenue = float(round(np.random.exponential(20), 2))
                    events.append({
                        "user_id": user,
                        "timestamp": (current_date + timedelta(minutes=int(np.random.randint(0, 1440)))),
                        "event_type": event_type,
                        "session_id": session_id,
                        "device": random.choice(DEVICES),
                        "experiment_variant": random.choice(EXPERIMENT_VARIANTS),
                        "revenue": revenue
                    })

    df = pd.DataFrame(events)
    # Ensure output dir exists
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/product_events.csv", index=False)
    print(f"Generated events: {len(df)} rows, saved to data/raw/product_events.csv")
    return df

if __name__ == "__main__":
    generate_events()