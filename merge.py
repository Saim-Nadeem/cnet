import pandas as pd
import glob

files = glob.glob("MachineLearningCVE/*.csv") 

dfs = []
for f in files:
    print("Loading:", f)
    dfs.append(pd.read_csv(f))

df = pd.concat(dfs, ignore_index=True)
df.to_csv("CICIDS2017.csv", index=False)

print("Merged CSV saved as CICIDS2017.csv")