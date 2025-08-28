import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pandas.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.cluster import KMeans
from pathlib import Path
import pandas as pd

BASE = Path(__file__).resolve().parent          # <- project folder
PROBES_DIR = BASE / "Probes"                    # adjust if you renamed it
DATA_DIR = BASE / "data"                        # or BASE if files sit at top-level


# -----------------------------
# Helper to map month to season
# -----------------------------
def get_season(month):
    if pd.isna(month):
        return pd.NA
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    else:
        return "Autumn"

# -----------------------------
# Small helper for trend lines
# -----------------------------
def _fit_time_trend(x_dates, y_vals):
    """
    Fit a simple linear trend y = a * t + b using numpy.polyfit, where t is matplotlib date number.
    Returns (t_sorted, yhat_sorted).
    """
    x = mdates.date2num(pd.to_datetime(x_dates))
    y = np.asarray(y_vals, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 2:
        return None, None
    a, b = np.polyfit(x, y, 1)
    order = np.argsort(x)
    return x[order], (a * x + b)[order]

# ------------------------------------
# Enhanced EDA (matplotlib-only)
# ------------------------------------
def enhanced_eda(df):
    print("\n=== ENHANCED EDA START ===")

    # Ensure season ordering for nicer plots
    if "season" in df.columns:
        season_order = ["Winter", "Spring", "Summer", "Autumn"]
        df["season"] = pd.Categorical(df["season"], categories=season_order, ordered=True)

    # Colors
    col_T1 = "tomato"
    col_T2 = "steelblue"
    col_T3 = "mediumpurple"
    col_moist = "royalblue"

    # 1) Seasonal boxplots for each feature (matplotlib only)
    features = ["T1 (soil T)", "T2 (surface T)", "T3 (air T)", "soil moisture count"]
    if "season" in df.columns:
        for col in [c for c in features if c in df.columns]:
            sub = df[["season", col]].dropna()
            if len(sub) == 0:
                continue

            order = [s for s in df["season"].cat.categories if s in sub["season"].unique()]
            grouped = [sub.loc[sub["season"] == s, col].values for s in order]

            plt.figure(figsize=(8, 5))
            plt.boxplot(
                grouped,
                labels=order,
                showfliers=True,           # keep outliers visible
                patch_artist=True,
                boxprops=dict(facecolor="#dbeafe", edgecolor="#1e3a8a"),
                medianprops=dict(color="#b91c1c"),
                whiskerprops=dict(color="#1e3a8a"),
                capprops=dict(color="#1e3a8a")
            )
            plt.title(f"{col} by Season — Boxplot")
            plt.xlabel("Season")
            plt.ylabel(col)
            plt.tight_layout()
            plt.show()

    # 2) Temperature trends over time (raw + linear regression lines)
    if "datetime_utc" in df.columns:
        plt.figure(figsize=(12, 6))
        for col, c in [("T1 (soil T)", col_T1), ("T2 (surface T)", col_T2), ("T3 (air T)", col_T3)]:
            if col not in df.columns:
                continue
            sub = df[["datetime_utc", col]].dropna().sort_values("datetime_utc")
            if len(sub) == 0:
                continue

            # faint raw series
            plt.plot(sub["datetime_utc"], sub[col], alpha=0.15, linewidth=1, color=c, label=f"{col} (raw)")

            # regression trend
            x_sorted, yhat_sorted = _fit_time_trend(sub["datetime_utc"], sub[col])
            if x_sorted is not None:
                plt.plot(mdates.num2date(x_sorted), yhat_sorted, color=c, linewidth=2.5, label=col)

        plt.title("Temperature Trends Over Time", fontsize=14, weight="bold")
        plt.ylabel("Temperature (°C)")
        plt.xlabel("Date")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        plt.show()

    # 3) Distributions (soil temp & soil moisture)
    plt.figure(figsize=(12, 5))

    if "T1 (soil T)" in df.columns:
        plt.subplot(1, 2, 1)
        vals = df["T1 (soil T)"].dropna()
        plt.hist(vals, bins=40, color="orange", alpha=0.85, edgecolor="black")
        plt.title("Soil Temperature Distribution", fontsize=12, weight="bold")
        plt.xlabel("T1 (soil T)")
        plt.ylabel("Count")
        plt.grid(alpha=0.2)

    if "soil moisture count" in df.columns:
        plt.subplot(1, 2, 2)
        vals = df["soil moisture count"].dropna()
        plt.hist(vals, bins=40, color=col_moist, alpha=0.85, edgecolor="black")
        plt.title("Soil Moisture Distribution", fontsize=12, weight="bold")
        plt.xlabel("Soil moisture count")
        plt.ylabel("Count")
        plt.grid(alpha=0.2)

    plt.tight_layout()
    plt.show()

    # 4) Correlation heatmap (sensors)
    corr_cols = [c for c in ["T1 (soil T)", "T2 (surface T)", "T3 (air T)", "soil moisture count"] if c in df.columns]
    if len(corr_cols) >= 2:
        corr_mat = df[corr_cols].dropna().corr()
        plt.figure(figsize=(6.6, 6))
        im = plt.imshow(corr_mat.values, vmin=-1, vmax=1, cmap="viridis")
        plt.colorbar(im, fraction=0.046, pad=0.04)
        plt.xticks(range(len(corr_cols)), corr_cols, rotation=45, ha="right")
        plt.yticks(range(len(corr_cols)), corr_cols)
        plt.title("Correlation Heatmap (Sensors)")
        # annotate cells
        for i in range(len(corr_cols)):
            for j in range(len(corr_cols)):
                val = corr_mat.values[i, j]
                plt.text(j, i, f"{val:.2f}", ha="center", va="center", color="white" if abs(val) > 0.6 else "black")
        plt.tight_layout()
        plt.show()

    # 5) Habitat × Season heatmap (mean T1)
    if {"habitat_type", "season"}.issubset(df.columns) and ("T1 (soil T)" in df.columns):
        pivot = (
            df.dropna(subset=["T1 (soil T)"])
              .pivot_table(index="habitat_type", columns="season", values="T1 (soil T)", aggfunc="mean")
              .sort_index()
        )
        if pivot.shape[0] > 0 and pivot.shape[1] > 0:
            plt.figure(figsize=(max(6, 0.8 * pivot.shape[1] + 5), max(6, 0.45 * pivot.shape[0] + 3)))
            im = plt.imshow(pivot.values, aspect="auto", cmap="viridis")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.xticks(range(pivot.shape[1]), pivot.columns.tolist(), rotation=45, ha="right")
            plt.yticks(range(pivot.shape[0]), [str(i) for i in pivot.index.tolist()])
            plt.title("Mean Soil Temperature (T1) — Habitat × Season")
            for i in range(pivot.shape[0]):
                for j in range(pivot.shape[1]):
                    v = pivot.values[i, j]
                    if pd.notna(v):
                        plt.text(j, i, f"{v:.2f}", ha="center", va="center", color="white" if v >= pivot.values.mean() else "black")
            plt.tight_layout()
            plt.show()

    # 6) Scatter matrix (pairwise relationships)
    if len(corr_cols) >= 3:
        clean = df[corr_cols].dropna()
        if len(clean) > 0:
            if len(clean) > 30000:  # cap for speed
                clean = clean.sample(30000, random_state=42)
            axarr = scatter_matrix(clean, figsize=(9, 9), diagonal="hist")
            for ax in axarr[-1, :]:
                ax.set_xlabel(ax.get_xlabel(), rotation=45, ha="right")
            for ax in axarr[:, 0]:
                ax.set_ylabel(ax.get_ylabel(), rotation=0, ha="right")
            plt.suptitle("Scatter Matrix — Sensor Readings", y=0.92)
            plt.tight_layout()
            plt.show()

    print("=== ENHANCED EDA COMPLETE ===\n")

# -----------------------------
# Main cleaning & merging
# -----------------------------
def clean_and_merge_probes(csv_folder, master_file_path, lower_bound=-10, upper_bound=60):
    # Gather all CSVs
    all_files = [os.path.join(csv_folder, f)
                 for f in os.listdir(csv_folder)
                 if f.lower().endswith(".csv")]
    dfs = []

    for file in all_files:
        try:
            temp_df = pd.read_csv(file, low_memory=False)

            # Extract probe ID and add column
            probe_id = os.path.basename(file).split("_")[1]
            temp_df["probe_id"] = str(probe_id)

            # Standardize and parse datetime
            temp_df.columns = temp_df.columns.str.strip()
            temp_df.rename(columns={"date and time in UTC": "datetime_utc"}, inplace=True)
            temp_df["datetime_utc"] = pd.to_datetime(temp_df["datetime_utc"], errors='coerce')

            # Clean numeric sensor columns
            for col in ["T1 (soil T)", "T2 (surface T)", "T3 (air T)", "soil moisture count"]:
                if col in temp_df.columns:
                    temp_df[col] = (
                        temp_df[col]
                        .astype(str).str.strip().replace("", pd.NA)
                        .pipe(pd.to_numeric, errors='coerce')
                    )
                    # Replace sentinel error values
                    temp_df.loc[temp_df[col] == -200, col] = pd.NA

            dfs.append(temp_df)
        except Exception as e:
            print(f"Failed to process {file}: {e}")

    # Concatenate & drop exact duplicates
    merged = pd.concat(dfs, ignore_index=True)
    merged.drop_duplicates(inplace=True)

    # -------------------------------
    # 🔵 HIGHLIGHT: Filter unrealistic temperature readings
    #     We drop any T below lower_bound or above upper_bound.
    #     Defaults: lower_bound = -10 °C, upper_bound = 60 °C
    # -------------------------------
    for c in ["T1 (soil T)", "T2 (surface T)", "T3 (air T)"]:
        if c not in merged.columns:
            merged[c] = pd.NA  # ensure column exists for masks

    lb, ub = lower_bound, upper_bound
    mask_bad = (
        (merged["T1 (soil T)"] < lb) | (merged["T1 (soil T)"] > ub) |
        (merged["T2 (surface T)"] < lb) | (merged["T2 (surface T)"] > ub) |
        (merged["T3 (air T)"] < lb) | (merged["T3 (air T)"] > ub)
    )
    print(f"Rows with unrealistic T (<{lb} or >{ub}) before filtering: {mask_bad.sum()}")
    print("Counts by probe for unrealistic T before filtering:")
    print(merged.loc[mask_bad, "probe_id"].value_counts(), "\n")

    merged = merged.loc[~mask_bad].copy()

    mask_bad_after = (
        (merged["T1 (soil T)"] < lb) | (merged["T1 (soil T)"] > ub) |
        (merged["T2 (surface T)"] < lb) | (merged["T2 (surface T)"] > ub) |
        (merged["T3 (air T)"] < lb) | (merged["T3 (air T)"] > ub)
    )
    print(f"Rows with unrealistic T after filtering: {mask_bad_after.sum()}\n")
    # -------------------------------

    # Diagnostic: count missing before dropping
    key_cols = ["datetime_utc", "T1 (soil T)", "T2 (surface T)", "T3 (air T)"]
    mask_missing = merged[key_cols].isna().any(axis=1)
    print(f"Rows with missing key values before drop: {mask_missing.sum()}")
    print("Counts by probe for missing key values before drop:")
    print(merged.loc[mask_missing, "probe_id"].value_counts(), "\n")

    # Drop any rows missing timestamp or any T-column
    before = len(merged)
    merged = merged.dropna(subset=key_cols)
    dropped = before - len(merged)
    print(f"Dropped {dropped} rows missing timestamp or temperature data\n")

    # Feature engineering: week, month, season
    merged["week"] = merged["datetime_utc"].dt.isocalendar().week
    merged["month"] = merged["datetime_utc"].dt.month
    merged["season"] = merged["month"].apply(get_season)

    # Load probe metadata and merge
    master_df = pd.read_excel(master_file_path, header=1)
    master_df.columns = master_df.columns.str.strip().str.lower()
    metadata_cols = [
        "tms probe number", "install date", "lat", "lon",
        "elevation (m)", "habitat type"
    ]
    probe_metadata = master_df[metadata_cols].copy()
    probe_metadata.columns = [
        "probe_id", "install_date", "latitude", "longitude",
        "elevation", "habitat_type"
    ]
    probe_metadata["probe_id"] = probe_metadata["probe_id"].astype(str)
    merged["probe_id"] = merged["probe_id"].astype(str)
    merged = merged.merge(probe_metadata, on="probe_id", how="left")

    return merged


if __name__ == "__main__":
    # Define paths
    csv_folder  = r"C:\Users\hirum\OneDrive\Desktop\Hiral_Mahida_DataScience\Probes"
    master_file = r"C:\Users\hirum\OneDrive\Desktop\Hiral_Mahida_DataScience\Gair Wood temp probe master records.xlsx"

    # Clean & merge
    df = clean_and_merge_probes(csv_folder, master_file, lower_bound=-10, upper_bound=60)

    # Quick EDA
    print("DataFrame info:")
    df.info()
    print("\nFirst five rows:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe())
    print("\nSeason counts:")
    print(df["season"].value_counts(), "\n")

    # Export cleaned snapshot
    out = r"C:\Users\hirum\OneDrive\Desktop\Hiral_Mahida_DataScience\cleaned_data.csv"
    df.to_csv(out, index=False)
    print("Cleaned data saved to", out)

    print("\n=== Cleaned dataset shape:", df.shape, "| Probes:", df['probe_id'].nunique(), "===\n")

    # ------------------------------------
    # ENHANCED EDA (with time trends)
    # ------------------------------------
    enhanced_eda(df)

    # ------------------------------------
    # 1) PCA on raw readings
    # ------------------------------------
    features = ["T1 (soil T)", "T2 (surface T)", "T3 (air T)", "soil moisture count"]
    df_pca = df.dropna(subset=[c for c in features if c in df.columns]).copy()

    if len(df_pca) >= 2:
        max_plot = 150_000
        if len(df_pca) > max_plot:
            df_pca = df_pca.sample(max_plot, random_state=42)

        use_cols = [c for c in features if c in df_pca.columns]
        Xs = StandardScaler().fit_transform(df_pca[use_cols])
        pca = PCA(n_components=2).fit(Xs)
        pcs = pca.transform(Xs)
        df_pca["PC1"], df_pca["PC2"] = pcs[:, 0], pcs[:, 1]

        print("\n[PCA] explained variance ratio:", np.round(pca.explained_variance_ratio_, 4))

        plt.figure(figsize=(8, 6))
        sc = plt.scatter(df_pca["PC1"], df_pca["PC2"], s=6, alpha=0.5,
                         c=df_pca[use_cols[0]], cmap="viridis")
        plt.colorbar(sc, label=use_cols[0])
        plt.title("PCA of Probe Readings")
        plt.xlabel("PC1"); plt.ylabel("PC2"); plt.tight_layout()
        plt.show()
    else:
        print("\n[PCA] Skipped — not enough rows with all numeric features.")

    # ------------------------------------
    # 2) Regression: predict T1 from T2, T3, soil moisture + season
    # ------------------------------------
    needed = ["T2 (surface T)", "T3 (air T)", "soil moisture count", "season", "T1 (soil T)"]
    need_present = [c for c in needed if c in df.columns]
    if set(needed).issubset(df.columns):
        df_reg = df.dropna(subset=needed).copy()
        if len(df_reg) >= 100:
            try:
                enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)  # sklearn >= 1.4
            except TypeError:
                enc = OneHotEncoder(handle_unknown="ignore", sparse=False)        # sklearn < 1.4

            season_ohe = enc.fit_transform(df_reg[["season"]])

            X_num = df_reg[["T2 (surface T)", "T3 (air T)", "soil moisture count"]].values
            X = np.hstack([X_num, season_ohe])
            y = df_reg["T1 (soil T)"].values

            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
            lr = LinearRegression().fit(Xtr, ytr)
            yhat = lr.predict(Xte)

            mae  = mean_absolute_error(yte, yhat)
            rmse = np.sqrt(mean_squared_error(yte, yhat))
            r2   = r2_score(yte, yhat)

            base = np.full_like(yte, ytr.mean())
            b_mae  = mean_absolute_error(yte, base)
            b_rmse = np.sqrt(mean_squared_error(yte, base))
            b_r2   = r2_score(yte, base)

            print("\n[Regression → Predict T1]")
            print(f"  MAE : {mae:.3f}   (baseline {b_mae:.3f})")
            print(f"  RMSE: {rmse:.3f}  (baseline {b_rmse:.3f})")
            print(f"  R²  : {r2:.3f}  (baseline {b_r2:.3f})")

            # Feature importance plot
            try:
                # Build feature names
                try:
                    season_feature_names = enc.get_feature_names_out(["season"]).tolist()
                except Exception:
                    cats = enc.categories_[0] if hasattr(enc, "categories_") else []
                    season_feature_names = [f"season_{c}" for c in cats]
                base_features = ["T2 (surface T)", "T3 (air T)", "soil moisture count"]
                feature_names = base_features + season_feature_names

                coefs = getattr(lr, "coef_", None)
                if coefs is not None and len(coefs) == len(feature_names):
                    importance = np.abs(coefs)
                    order = np.argsort(importance)

                    plt.figure(figsize=(8, max(3, 0.4 * len(feature_names))))
                    plt.barh(np.array(feature_names)[order], importance[order])
                    plt.title("Feature Importance (|Linear Regression Coefficients|) — Predicting T1")
                    plt.xlabel("|Coefficient|")
                    plt.tight_layout()
                    plt.show()
                else:
                    print("\n[Feature Importance] Skipped — mismatch in feature names/coefficients.")
            except Exception as e:
                print(f"\n[Feature Importance] Error: {e}")

        else:
            print("\n[Regression] Skipped — need ≥100 rows with T2/T3/soil moisture/season/T1.")
    else:
        print(f"\n[Regression] Skipped — missing columns: {set(needed) - set(need_present)}")

    # ------------------------------------
    # 3) Clustering: per-probe averages (KMeans)
    # ------------------------------------
    have_cols = [c for c in features if c in df.columns]
    if len(have_cols) >= 2:
        df_means = df.dropna(subset=have_cols).groupby("probe_id")[have_cols].mean()
        if len(df_means) >= 3:
            Xc = StandardScaler().fit_transform(df_means)
            km = KMeans(n_clusters=3, random_state=42, n_init=10)
            df_means["cluster"] = km.fit_predict(Xc)
            sil = silhouette_score(Xc, df_means["cluster"]) if len(df_means) > 3 else np.nan
            if not np.isnan(sil):
                print(f"\n[Clustering] KMeans (k=3) — silhouette: {sil:.3f}")
            else:
                print("\n[Clustering] KMeans (k=3)")
            print("\nFirst 10 probes with clusters:\n", df_means.head(10))

            # 2D visualization with first two features
            xcol, ycol = have_cols[0], have_cols[1]
            plt.figure(figsize=(8, 6))
            plt.scatter(df_means[xcol], df_means[ycol],
                        c=df_means["cluster"], cmap="tab10", s=120, edgecolor="k", alpha=0.9)
            plt.xlabel(f"Avg {xcol}"); plt.ylabel(f"Avg {ycol}")
            plt.title("Probe Clusters by Average Conditions (k=3)")
            plt.tight_layout()
            plt.show()
        else:
            print("\n[Clustering] Skipped — need ≥3 probes with averages.")
    else:
        print("\n[Clustering] Skipped — not enough numeric features.")
