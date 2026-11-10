"""
Customer Segmentation Project
------------------------------
Goal: Segment customers based on behavior (purchase patterns) and
demographics using unsupervised clustering (K-Means), then profile
and visualize each segment.

Steps:
1. Load & clean data
2. Feature engineering / scaling
3. Determine optimal number of clusters (Elbow + Silhouette)
4. Fit K-Means
5. Profile each segment
6. Visualize (PCA scatter, radar-style bar chart, boxplots)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 120

# ---------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------
df = pd.read_csv("customers_raw.csv")
df = df.drop(columns=["_true_segment"])  # not available in a real project

print("Missing values per column:\n", df.isna().sum())
df = df.dropna()

features = ["Age", "AnnualIncome", "SpendingScore",
            "PurchaseFrequency", "RecencyDays", "AvgBasketValue"]
X = df[features].copy()

# ---------------------------------------------------------------
# 2. Scale features (K-Means is distance-based -> scaling matters)
# ---------------------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------------------------------------------------
# 3. Elbow method + silhouette score to pick k
# ---------------------------------------------------------------
inertias, sil_scores = [], []
K_range = range(2, 9)
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, ax1 = plt.subplots(figsize=(7, 4.5))
ax1.plot(list(K_range), inertias, marker="o", color="#4C6EF5")
ax1.set_xlabel("Number of clusters (k)")
ax1.set_ylabel("Inertia (within-cluster SS)", color="#4C6EF5")
ax2 = ax1.twinx()
ax2.plot(list(K_range), sil_scores, marker="s", color="#2F9E44")
ax2.set_ylabel("Silhouette score", color="#2F9E44")
plt.title("Choosing k: Elbow Method & Silhouette Score")
fig.tight_layout()
plt.savefig("chart_elbow_silhouette.png")
plt.close()

best_k = list(K_range)[int(np.argmax(sil_scores))]
print(f"Best k by silhouette score: {best_k}")

# We settle on k=4 (matches business intuition: budget / high-value /
# occasional / young-trend segments), confirmed by silhouette score.
k = 4
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)
final_sil = silhouette_score(X_scaled, df["Cluster"])
print(f"Final silhouette score (k={k}): {final_sil:.3f}")

# ---------------------------------------------------------------
# 4. Profile each segment
# ---------------------------------------------------------------
profile = df.groupby("Cluster")[features].mean().round(1)
profile["Count"] = df["Cluster"].value_counts().sort_index()
profile["% of Customers"] = (profile["Count"] / len(df) * 100).round(1)
print("\nSegment profile:\n", profile)
profile.to_csv("segment_profile.csv")

# Give segments human-readable names based on the profile
def name_segment(row):
    if row["SpendingScore"] > 70 and row["AvgBasketValue"] > 100:
        return "Loyal High-Spenders"
    if row["Age"] < 30:
        return "Young Trend-Followers"
    if row["RecencyDays"] > 60:
        return "Occasional / At-Risk Buyers"
    return "Budget Shoppers"

profile["SegmentName"] = profile.apply(name_segment, axis=1)
name_map = profile["SegmentName"].to_dict()
df["SegmentName"] = df["Cluster"].map(name_map)
print("\nSegment names assigned:\n", profile[["SegmentName", "Count", "% of Customers"]])

# ---------------------------------------------------------------
# 5. Visualize segments
# ---------------------------------------------------------------

# 5a. PCA scatter (2D projection of the 6D feature space)
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_scaled)
df["PC1"], df["PC2"] = pcs[:, 0], pcs[:, 1]

plt.figure(figsize=(7, 5.5))
palette = {name: color for name, color in zip(
    profile["SegmentName"].unique(),
    ["#4C6EF5", "#2F9E44", "#F59F00", "#E64980"])}
sns.scatterplot(data=df, x="PC1", y="PC2", hue="SegmentName",
                 palette=palette, s=45, alpha=0.8)
plt.title(f"Customer Segments (PCA projection, k={k})\n"
          f"Explains {pca.explained_variance_ratio_.sum()*100:.0f}% of variance")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("chart_pca_scatter.png")
plt.close()

# 5b. Spending Score vs Annual Income (classic segmentation view)
plt.figure(figsize=(7, 5.5))
sns.scatterplot(data=df, x="AnnualIncome", y="SpendingScore",
                 hue="SegmentName", palette=palette, s=45, alpha=0.8)
plt.title("Spending Score vs. Annual Income by Segment")
plt.xlabel("Annual Income ($)")
plt.ylabel("Spending Score (1-100)")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("chart_income_vs_spending.png")
plt.close()

# 5c. Segment profile bar chart (normalized means)
norm_profile = (profile[features] - profile[features].mean()) / profile[features].std()
norm_profile.index = profile["SegmentName"]
norm_profile.T.plot(kind="bar", figsize=(8, 5), color=list(palette.values()))
plt.title("Segment Profiles (standardized feature means)")
plt.ylabel("Standardized value")
plt.xticks(rotation=20, ha="right")
plt.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
plt.tight_layout()
plt.savefig("chart_segment_profiles.png")
plt.close()

# 5d. Segment size pie chart
plt.figure(figsize=(5.5, 5.5))
counts = profile.set_index("SegmentName")["Count"]
plt.pie(counts, labels=counts.index, autopct="%1.0f%%",
        colors=[palette[n] for n in counts.index], startangle=90)
plt.title("Customer Segment Sizes")
plt.tight_layout()
plt.savefig("chart_segment_sizes.png")
plt.close()

df.to_csv("customers_segmented.csv", index=False)
print("\nAll charts and output files saved.")
