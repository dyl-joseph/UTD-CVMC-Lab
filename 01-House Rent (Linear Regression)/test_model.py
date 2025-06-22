import pandas as pd
import matplotlib.pyplot as plt


x = pd.read_csv('01-House Rent (Linear Regression)/sqft.csv')
y = pd.read_csv('01-House Rent (Linear Regression)/price.csv')
x = x['sqfeet']
y = y['price']
training_size = int(0.8*x.size)

# parameters
a = 0
b = 0
with open('01-House Rent (Linear Regression)/parameter_weights.txt') as f:
    f.readline() # skips the epoch line

    a = float(f.readline().strip('a: '))  # grabs value associated with a 
    b = float(f.readline().strip('b: ')) # grabs value associated with b

# test model
y_hat = (a*x[training_size+1:]+b)*x.max()
y[training_size+1:] *= y.max()
print(f'error: {(y[training_size+1:]-y_hat)}')

plt.plot(x[training_size+1:],y[training_size+1:],'o')
plt.plot(x[training_size+1:], ((a*x[training_size+1:])+b), 'r-')
plt.savefig('model.png')

plt.show()