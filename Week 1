
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "water_data.csv"
OUTPUT_FILE = "cleaned_water_data.csv"

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Place your dataset in the project root.")
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.fillna(method="ffill").fillna(method="bfill")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        lo, hi = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lo, hi)
    return df

def eda(df: pd.DataFrame) -> None:
    for col in ["rainfall", "temperature", "consumption", "reservoir_level"]:
        plt.figure(figsize=(10,4))
        plt.plot(df["date"], df[col])
        plt.title(f"Time Series: {col}")
        plt.xlabel("date")
        plt.ylabel(col)
        plt.tight_layout()
        plt.savefig(f"eda_timeseries_{col}.png")
        plt.close()

    corr = df[["rainfall","temperature","consumption","reservoir_level"]].corr()
    fig, ax = plt.subplots(figsize=(6,5))
    cax = ax.imshow(corr, interpolation="nearest")
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticklabels(corr.columns)
    for (i, j), val in np.ndenumerate(corr.values):
        ax.text(j, i, f"{val:.2f}", ha="center", va="center")
    fig.colorbar(cax)
    plt.title("Feature Correlation")
    plt.tight_layout()
    plt.savefig("eda_correlation.png")
    plt.close()

if __name__ == "__main__":
    print(" Loading data...")
    df = load_data(INPUT_FILE)
    print(df.head())

    print("\n Preprocessing...")
    df_clean = preprocess(df)
    print(df_clean.describe())

    print("\n EDA plots...")
    eda(df_clean)

    print("\n Saving cleaned dataset...")
    df_clean.to_csv(OUTPUT_FILE, index=False)
    print(f" Saved {OUTPUT_FILE} and EDA images.")
