"""
Generates a synthetic but realistic customer dataset for the
Customer Segmentation Project.
"""
import numpy as np
import pandas as pd

np.random.seed(42)
segment_sizes = [150, 180, 140, 130]
segment_names = ["Budget Shopper", "Loyal High-Spender", "Occasional Buyer", "Young Trend-Follower"]

rows = []
cust_id = 1000
for seg_idx, size in enumerate(segment_sizes):
    for _ in range(size):
        if seg_idx == 0:
            age = np.random.normal(52, 9); income = np.random.normal(35000, 6000)
            spending_score = np.random.normal(30, 10); purchase_freq = np.random.normal(10, 3)
            recency = np.random.normal(20, 10); avg_basket = np.random.normal(25, 8)
        elif seg_idx == 1:
            age = np.random.normal(42, 8); income = np.random.normal(95000, 15000)
            spending_score = np.random.normal(82, 8); purchase_freq = np.random.normal(22, 4)
            recency = np.random.normal(8, 5); avg_basket = np.random.normal(140, 30)
        elif seg_idx == 2:
            age = np.random.normal(48, 12); income = np.random.normal(60000, 12000)
            spending_score = np.random.normal(45, 12); purchase_freq = np.random.normal(4, 2)
            recency = np.random.normal(75, 25); avg_basket = np.random.normal(55, 15)
        else:
            age = np.random.normal(24, 4); income = np.random.normal(42000, 9000)
            spending_score = np.random.normal(68, 10); purchase_freq = np.random.normal(15, 4)
            recency = np.random.normal(15, 8); avg_basket = np.random.normal(48, 12)

        rows.append({
            "CustomerID": cust_id,
            "Age": max(18, round(age)),
            "AnnualIncome": max(12000, round(income, -2)),
            "SpendingScore": min(100, max(1, round(spending_score))),
            "PurchaseFrequency": max(1, round(purchase_freq)),
            "RecencyDays": max(1, round(recency)),
            "AvgBasketValue": max(5, round(avg_basket, 2)),
            "Gender": np.random.choice(["Male", "Female"]),
            "_true_segment": segment_names[seg_idx],
        })
        cust_id += 1

df = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)
df.to_csv("customers_raw.csv", index=False)
print(df.shape)
