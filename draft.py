import pandas as pd
import numpy as np

data = np.array([[1e30, 1, 2],
                 [3, 1e30, 4],
                 [5, 6, 7]])
df = pd.DataFrame(data, columns=['a', 'b', 'c'])
print(df)
df.mask(df == 1e30, inplace=True)
df.dropna(axis=0, how='any', inplace=True)
df.reset_index(drop=True, inplace=True)
print(df)
