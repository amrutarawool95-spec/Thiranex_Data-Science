"""
Data Cleaning & Visualization Project
Raw (messy) retail sales data -> cleaned data -> dashboard + insights.
To use your own file: set RAW_FILE = "your.csv" and skip make_raw_data().
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns

RAW_FILE = "raw_sales.csv"
rng = np.random.default_rng(42)

# ---------- 0. Build a realistic messy dataset ----------
def make_raw_data(n=1200):
    cats = {"Electronics": (150, 900), "Clothing": (15, 120), "Home": (20, 250), "Beauty": (8, 70), "Sports": (20, 200)}
    regions = ["North", "South", "East", "West"]
    dates = pd.to_datetime("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    cat = rng.choice(list(cats), n, p=[.2, .3, .2, .15, .15])
    price = np.array([rng.uniform(*cats[c]) for c in cat]).round(2)
    qty = rng.integers(1, 6, n)
    df = pd.DataFrame({
        "order_id": np.arange(1001, 1001 + n), "order_date": dates, "customer_name": rng.choice(
            ["Asha Patil", "Rohan Mehta", "Sneha Kulkarni", "Vikram Singh", "Priya Nair", "Arjun Rao", "Meera Joshi"], n),
        "category": cat, "region": rng.choice(regions, n), "unit_price": price, "quantity": qty,
        "rating": rng.choice([1, 2, 3, 4, 5], n, p=[.05, .1, .2, .35, .3]).astype(float)})
    # --- inject mess ---
    df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d")
    idx = rng.choice(n, 150, replace=False); df.loc[idx, "order_date"] = pd.to_datetime(df.loc[idx, "order_date"]).dt.strftime("%d/%m/%Y")
    df["category"] = df["category"].map(lambda x: rng.choice([x, x.upper(), x.lower(), " " + x + " "], p=[.7, .1, .1, .1]))
    df["region"] = df["region"].map(lambda x: rng.choice([x, x.lower(), x + " "], p=[.8, .1, .1]))
    df.loc[rng.choice(n, 90, replace=False), "unit_price"] = np.nan
    df.loc[rng.choice(n, 70, replace=False), "rating"] = np.nan
    df.loc[rng.choice(n, 40, replace=False), "customer_name"] = np.nan
    df.loc[rng.choice(n, 12, replace=False), "unit_price"] *= 25      # outliers
    df.loc[rng.choice(n, 5, replace=False), "quantity"] = -3           # invalid
    df.loc[rng.choice(n, 6, replace=False), "quantity"] = 250          # outliers
    df = pd.concat([df, df.sample(60, random_state=1)], ignore_index=True)  # duplicates
    df["unit_price"] = df["unit_price"].astype("object")
    df.loc[rng.choice(len(df), 25, replace=False), "unit_price"] = "$" + df["unit_price"].astype(str)  # text prices
    df.sample(frac=1, random_state=3).to_csv(RAW_FILE, index=False)

make_raw_data()

# ---------- 1. Inspect ----------
raw = pd.read_csv(RAW_FILE)
log = {"rows_raw": len(raw)}
print(raw.info()); print(raw.isna().sum())

# ---------- 2. Clean ----------
df = raw.copy()
# a) duplicates
log["duplicates_removed"] = int(df.duplicated().sum()); df = df.drop_duplicates()
# b) text standardisation
for c in ["category", "region"]: df[c] = df[c].str.strip().str.title()
# c) fix data types (prices with '$', mixed date formats)
df["unit_price"] = pd.to_numeric(df["unit_price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
d1 = pd.to_datetime(df["order_date"], format="%Y-%m-%d", errors="coerce")
d2 = pd.to_datetime(df["order_date"], format="%d/%m/%Y", errors="coerce")
df["order_date"] = d1.fillna(d2)
# d) invalid values
log["invalid_qty_removed"] = int((df["quantity"] <= 0).sum()); df = df[df["quantity"] > 0]
# e) outliers: IQR rule *within each category* (cap prices, drop absurd quantities)
def iqr_bounds(s): q1, q3 = s.quantile([.25, .75]); i = q3 - q1; return q1 - 1.5 * i, q3 + 1.5 * i
out_price = 0
for c, g in df.groupby("category"):
    lo, hi = iqr_bounds(g["unit_price"].dropna())
    m = (df["category"] == c) & (df["unit_price"] > hi); out_price += int(m.sum()); df.loc[m, "unit_price"] = np.nan  # re-impute below
log["price_outliers_fixed"] = out_price
lo, hi = iqr_bounds(df["quantity"]); log["qty_outliers_removed"] = int((df["quantity"] > hi).sum()); df = df[df["quantity"] <= hi]
# f) missing values
log["missing_price_imputed"] = int(df["unit_price"].isna().sum())
df["unit_price"] = df["unit_price"].fillna(df.groupby("category")["unit_price"].transform("median"))   # median by category
log["missing_rating_imputed"] = int(df["rating"].isna().sum()); df["rating"] = df["rating"].fillna(df["rating"].median())
log["missing_name_labelled"] = int(df["customer_name"].isna().sum()); df["customer_name"] = df["customer_name"].fillna("Unknown")
df = df.dropna(subset=["order_date"])
# g) feature engineering
df["revenue"] = (df["unit_price"] * df["quantity"]).round(2)
df["month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
df = df.sort_values("order_date").reset_index(drop=True)
log["rows_clean"] = len(df)
df.to_csv("cleaned_sales.csv", index=False)
print(pd.Series(log))

# ---------- 3. Visualise ----------
sns.set_theme(style="whitegrid", font_scale=1.0)
pal = sns.color_palette("viridis", 5)
fig = plt.figure(figsize=(17, 14)); gs = fig.add_gridspec(3, 3, hspace=.45, wspace=.3)
fig.suptitle("Retail Sales Dashboard — 2025", fontsize=22, fontweight="bold", y=.97)

# KPI row
kpis = [("Total revenue", f"${df.revenue.sum():,.0f}"), ("Orders", f"{len(df):,}"),
        ("Avg order value", f"${df.revenue.mean():,.0f}"), ("Avg rating", f"{df.rating.mean():.2f} / 5")]
kax = fig.add_subplot(gs[0, :]); kax.axis("off")
for i, (k, v) in enumerate(kpis):
    kax.text(.125 + i * .25, .6, v, ha="center", fontsize=26, fontweight="bold", color="#2a4d8f")
    kax.text(.125 + i * .25, .15, k, ha="center", fontsize=13, color="gray")

ax = fig.add_subplot(gs[1, 0:2]); m = df.groupby("month").revenue.sum()
ax.plot(m.index, m.values, marker="o", color="#2a4d8f", lw=2.5); ax.fill_between(m.index, m.values, alpha=.15, color="#2a4d8f")
ax.set_title("Monthly revenue trend", fontweight="bold"); ax.set_ylabel("Revenue ($)"); ax.tick_params(axis="x", rotation=45)

ax = fig.add_subplot(gs[1, 2]); c = df.groupby("category").revenue.sum().sort_values()
ax.barh(c.index, c.values, color=pal); ax.set_title("Revenue by category", fontweight="bold")

ax = fig.add_subplot(gs[2, 0]); r = df.groupby("region").revenue.sum().sort_values(ascending=False)
ax.pie(r, labels=r.index, autopct="%1.0f%%", colors=sns.color_palette("Set2"), startangle=90); ax.set_title("Revenue share by region", fontweight="bold")

ax = fig.add_subplot(gs[2, 1]); sns.boxplot(data=df, x="category", y="unit_price", hue="category", palette="viridis", legend=False, ax=ax)
ax.set_title("Price spread per category (after outlier fix)", fontweight="bold"); ax.tick_params(axis="x", rotation=30); ax.set_xlabel("")

ax = fig.add_subplot(gs[2, 2]); h = df.pivot_table(index="category", columns="region", values="rating", aggfunc="mean")
sns.heatmap(h, annot=True, fmt=".2f", cmap="YlGnBu", ax=ax, cbar=False); ax.set_title("Avg rating: category × region", fontweight="bold"); ax.set_xlabel(""); ax.set_ylabel("")
fig.savefig("dashboard.png", dpi=150, bbox_inches="tight")

# before/after cleaning chart
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
rawp = pd.to_numeric(raw["unit_price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce").dropna()
sns.boxplot(x=rawp, ax=ax[0], color="#e07b7b"); ax[0].set_title("Unit price BEFORE cleaning (outliers)", fontweight="bold")
sns.boxplot(x=df["unit_price"], ax=ax[1], color="#6fbf8f"); ax[1].set_title("Unit price AFTER cleaning", fontweight="bold")
fig.savefig("before_after.png", dpi=150, bbox_inches="tight")

# ---------- 4. Insights ----------
top_c = c.idxmax(); top_r = r.idxmax(); best_m = m.idxmax()
print(f"\nTop category: {top_c} ({c.max()/c.sum():.0%} of revenue)\nTop region: {top_r}\nBest month: {best_m:%B}")
with open("cleaning_log.txt", "w") as f:
    f.write("\n".join(f"{k}: {v}" for k, v in log.items()))
    f.write(f"\n\nTop category: {top_c} ({c.max()/c.sum():.0%})\nTop region: {top_r}\nBest month: {best_m:%B %Y}\n")
                                                                                                                               
