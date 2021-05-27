import pandas as pd

data = {'Vegetables': ['Apple', 'Orange', 'Banana', 'Coconut', 'Mango']}
df = pd.DataFrame(data, columns=['Vegetables'])

df.rename(columns={'Vegetables': 'Fruits'},inplace=True)

print(df)
