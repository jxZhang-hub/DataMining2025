import dask.dataframe as dd
import numpy as np

ddf = dd.read_parquet('./10G_data_new/part-00004.parquet', chunksize='1000MB')
