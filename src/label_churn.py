
import pandas as pd
import numpy as np
from datetime import timedelta
import os

def generate_churn_label(user_df, multiplier=0.6):
    """
    Generates churn labels based on engagement signals in user_df.
    multiplier controls overall churn rate (0.0 - 1.0).
    """
    # Use behavioral signals to compute an engagement score
    # Weights chosen to favor events per session (engagement) and revenue
    engagement_score = (
        (user_df["avg_events_per_session"].fillna(0) * 0.4) +
        (user_df["revenue_per_session"].fillna(0) * 0.3) +
        (user_df["total_sessions"].fillna(0) * 0.3)
    )

    # Normalize to 0..1
    if engagement_score.max() == engagement_score.min():
        engagement_norm = np.zeros_like(engagement_score)
    else:
        engagement_norm = (engagement_score - engagement_score.min()) / (engagement_score.max() - engagement_score.min())

    # Lower engagement -> higher churn probability
    churn_prob = 1.0 - engagement_norm
    churn_prob = churn_prob * multiplier  # control global churn rate

    # Draw random outcomes
    rng = np.random.default_rng(42)
    churn_labels = rng.random(len(churn_prob)) < churn_prob
    return churn_labels.astype(bool)

def define_churn(user_table_path="data/clean/user_dataset.csv",
                 events_path="data/raw/product_events.csv",
                 out_path="data/analytics/user_with_churn.csv"):
    user_df = pd.read_csv(user_table_path, parse_dates=["first_active", "last_active"])
    # (Optionally use events_path to compute churn window - not required here)
    # Generate churn labels
    user_df["churned"] = generate_churn_label(user_df, multiplier=0.6)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    user_df.to_csv(out_path, index=False)
    print(f"Churn labels written to {out_path}. Churn distribution:\n{user_df['churned'].value_counts()}")
    return user_df

if __name__ == "__main__":
    define_churn()