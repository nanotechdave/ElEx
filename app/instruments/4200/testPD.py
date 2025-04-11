import pandas as pd

df1 = pd.DataFrame(columns=['A', 'B', 'C'])
print(df1)

is_empty = df1.iloc[:, 0].isna().all()
print(is_empty)