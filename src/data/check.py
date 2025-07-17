import pandas as pd

df = pd.read_parquet("data/interim/porto_traj_portobox_30s.parquet")
print("rows:", len(df))
print(df.head())

print("\ncolumns:", df.columns.tolist())

# sampling rate check per trip (random 1 trip)
tid = df.trip_id.sample(1, random_state=0).iloc[0]
g = df[df.trip_id == tid].sort_values("timestamp")
deltas = g.timestamp.diff().dt.total_seconds().dropna()
print(f"median dt for trip {tid}:", deltas.median())
print(f"unique dt counts:\n{deltas.value_counts().head()}")