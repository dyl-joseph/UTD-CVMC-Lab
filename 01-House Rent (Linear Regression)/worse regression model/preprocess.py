import pandas as pd

df = pd.read_csv('01-House Rent (Linear Regression)/housing_train.csv')

x = df['sqfeet']
x = x.astype(float)

y = df['price']
y = y.astype(float)

# normalizes data in range [-1,1]
for i in range (x.size):
    x[i] = x[i]/x.max()
    y[i] = y[i]/y.max()

x.to_csv('01-House Rent (Linear Regression)/sqft.csv')
y.to_csv('01-House Rent (Linear Regression)/price.csv')

print('finished')