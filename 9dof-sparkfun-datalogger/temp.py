import pandas as pd

# Create a sample DataFrame
data = {'Name': ['Electronics', 'Clothing', 'E', 'Electronics', 'Clothing', 'Pasta'],'Score': [1000, 25, 200, 800, 50, 300]}
df = pd.DataFrame(data)
df
aggregation_functions = {'Score':'sum','Name':'first'}
df.aggregate(aggregation_functions)


df.sort_values(['Category','Score'])

print(df)