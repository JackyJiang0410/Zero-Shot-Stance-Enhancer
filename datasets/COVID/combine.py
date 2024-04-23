import pandas as pd

fm = pd.read_csv("home.csv")
hc = pd.read_csv("masks.csv")
la = pd.read_csv("school_closures.csv")
la2 = pd.read_csv("fauci.csv")

df = pd.concat([fm, hc, la, la2])
df.reset_index(drop=True)

df = df[['Tweet', 'background', 'Target 1', 'Stance 1']]
df = df.rename(columns={'Tweet': 'text', 'Target 1': 'target', 'Stance 1': 'label'})

df.to_csv("covid19_background.csv", index=False, encoding='utf-8')
print(df.target.drop_duplicates(keep='first'))
