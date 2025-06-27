import pandas as pd

df = pd.read_csv("experiments.tsv", sep="\t")
df["USG_group"] = df["USG"].round(1)
print(df["USG_group"].unique())
print(df["USG_group"].value_counts())