import pandas as pd
import numpy as np

def load_and_clean_data(filepath):
    df = pd.read_csv(filepath)
    df = df.drop("Reply", axis=1)
    df = df.dropna(subset=['Review'])
    df["Rating"] = np.where(df["Rating"] > 3, 1, 0)
    df["Sentiment"] = df["Rating"].replace({1: 'positive', 0: 'negative'})
    return df