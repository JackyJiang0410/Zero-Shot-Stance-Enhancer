import pandas as pd

fm = pd.read_csv("Feminist_Movement.csv")
hc = pd.read_csv("Hillary_Clinton.csv")
la = pd.read_csv("Legalization_of_Abortion.csv")
df = pd.concat([fm, hc, la])
df.reset_index()

df.to_csv("semeval16_background.csv", index=False, encoding='utf-8')
print(df.target.drop_duplicates(keep='first'))
