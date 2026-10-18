import os
import pandas as pd

os.chdir('c:/Users/MSUSERSL123/Documents/maersk/supply-chain-platform')
from src.ml.features import build_delay_target

df = pd.read_csv('data/raw/shipments.csv')
y = build_delay_target(df)
print(y.value_counts(dropna=False).to_dict())
print('valid_rows', int(y.notna().sum()))
print('class_counts', y.dropna().value_counts().to_dict())
