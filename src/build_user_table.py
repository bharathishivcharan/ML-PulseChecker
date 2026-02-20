import pandas as pd
import os

def build_user_table(raw_path="data/raw/product_events.csv", out_path="data/clean/user_dataset.csv"):
    df = pd.read_csv(raw_path, parse_dates=["timestamp"])
    # Basic user aggregations
    user_df = df.groupby("user_id").agg(
        total_events=("event_type", "count"),
        total_sessions=("session_id", "nunique"),
        total_revenue=("revenue", "sum"),
        first_active=("timestamp", "min"),
        last_active=("timestamp", "max")
    ).reset_index()

    # Derived features
    user_df["active_days"] = (user_df["last_active"] - user_df["first_active"]).dt.days + 1
    user_df["avg_events_per_session"] = (user_df["total_events"] / user_df["total_sessions"]).fillna(0)
    user_df["revenue_per_session"] = (user_df["total_revenue"] / user_df["total_sessions"]).fillna(0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    user_df.to_csv(out_path, index=False)
    print(f"Built user table ({len(user_df)} users) -> {out_path}")
    return user_df

if __name__ == "__main__":
    build_user_table()