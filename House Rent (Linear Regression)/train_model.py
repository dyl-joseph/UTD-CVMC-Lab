import pandas as pd
import matplotlib.pyplot as plt


x = pd.read_csv('/Users/dylanjoseph/Library/CloudStorage/OneDrive-Personal/Code/CVMC Lab Work/House Rent (Linear Regression)/sqft.csv')
y = pd.read_csv('/Users/dylanjoseph/Library/CloudStorage/OneDrive-Personal/Code/CVMC Lab Work/House Rent (Linear Regression)/price.csv')
x = x['sqfeet']
y = y['price']
training_size = int(0.8*x.size)

lr = 1e-8

# parameters
a = 1
b = 0


# train
gradient_a = 0
gradient_b = 0
epochs = 750

for epoch in range(epochs):
    for i in range(0,training_size,10):
        for n in range(i,i+10):
            if(n>training_size-1):
                break
            loss = (a*x[n]+b-y[n])
            gradient_a += (loss)*2*x[n]
            gradient_b += (loss)*2


        a = a - (lr*((1/10)*(gradient_a)))
        b = b - (lr*((1/10)*(gradient_b)))
        gradient_a = 0    
        gradient_b = 0
    print(f'a: {a}, b: {b}')


# test model

y_hat = (a*x[training_size+1:]+b)*x.max
y[training_size+1:] *= y.max
print(f'error: {(y[training_size+1:]-y_hat)}')

plt.plot(x[training_size+1:],y[training_size+1:],'o')
plt.plot(x[training_size+1:], ((a*x[training_size+1:])+b), 'r-')
plt.show()

# save values into a .txt file
file = open('/Users/dylanjoseph/Library/CloudStorage/OneDrive-Personal/Code/CVMC Lab Work/House Rent (Linear Regression)/parameter_weights.txt', 'w')
file.write(f'epochs: {epochs} \na: {a} \nb: {b}')