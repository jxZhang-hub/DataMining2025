import os
import pandas as pd


folder_path = './10G_data_new/'

parquet_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.parquet')]

if not parquet_files:
    print("No parquet file found.")
else:
    print(f"{len(parquet_files)} parquet files found\n")

for idx, file in enumerate(parquet_files[3:5]):
    print(f"file {idx+1}: {os.path.basename(file)}")
    try:
        df = pd.read_parquet(file)
        print("data type:\n", df.dtypes)
        print("preview:\n", df.iloc[0])
    except Exception as e:
        print(f"Fails to read:{e}")
    print("\n" + "-"*80 + "\n")