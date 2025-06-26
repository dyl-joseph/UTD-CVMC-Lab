import pandas as pd
import matplotlib.pyplot as plt
import random, copy

main_x = pd.read_csv('sqft.csv')
main_y = pd.read_csv('price.csv')

x = main_x['sqfeet']
y = main_y['price']
training_size = int(0.8*x.size) #[0,212151] ~ 80%
validation_size = int(0.9*x.size) #[212152,238671] ~ 10%

x = x[:training_size] # split and get training data
y = y[:training_size]


val_x = main_x['sqfeet'] # split and get validation data
val_x = val_x[training_size:validation_size]
val_x = val_x.reset_index(drop=True)

val_y = main_y['price']
val_y = val_y[training_size:validation_size]
val_y = val_y.reset_index(drop=True)


test_x = main_x['sqfeet'] # split and get test data 
test_x = test_x[validation_size:]
test_x = test_x.reset_index(drop=True)

test_y = main_y['price']
test_y = test_y[validation_size:]
test_y = test_y.reset_index(drop=True)


# parameters
a = random.random()
b = random.random()

# output + calculate initial values of line
print(f'initial values: a: {a}, b: {b}')
pred_y = [(a*sqft+b) for sqft in test_x]
plt.figure()
plt.subplot(1,2,1)
plt.title('line of best fit')
plt.plot(test_x, pred_y, label='initial line', color = 'red')

# train
lr = 1e-2
gradient_a = 0
gradient_b = 0
epochs = 1500
val_check = 20 # utilizes validation every 20 epochs

train_loss_lst = []
val_loss_lst = []

def compute_loss(pred_y,y):
    total_loss = 0
    for i in range(len(y)):
        total_loss += (1/training_size) * ((pred_y[i]-y[i])**2)

    return total_loss
def get_grad_a(pred, x, y):
    grad_a = 0
    for i in range(len(pred)):
        grad_a += (1/training_size)*((pred[i]-y[i])*(2*x[i]))
    return grad_a
def get_grad_b(pred,y):
    grad_b = 0
    for i in range(len(pred)):
        grad_b += (1/training_size) * ((pred[i]-y[i])*2)
    return grad_b
def shuffle(x,y):
    # combine data into one dataframe
    data = pd.DataFrame({'x': x, 'y':y})
    data = data.sample(frac=1).reset_index(drop=True)
    
    #seperate data
    x = data['x']
    y = data['y']

    return x,y
for epoch in range(epochs):
    x,y = shuffle(x,y)
    if (epoch%val_check==val_check-1):
        pred_val_y = [a*val_x[i]+b for i in range(len(val_x))]
        val_loss = compute_loss(pred_val_y, val_y)
        val_loss_lst.append(val_loss)
        print(f'epoch: {epoch}  val loss: {val_loss}')

    pred_y = [a*x[i]+b for i in range(training_size)]
    loss = compute_loss(pred_y,y)
    train_loss_lst.append(loss)
    # print(f'loss: {loss}')
    
    grad_a = get_grad_a(pred_y,x,y)
    grad_b = get_grad_b(pred_y,y)

    a -= (lr*(grad_a))
    b -= (lr*(grad_b))
    
    # print(f'a: {a}, b: {b}')


# test model

pred_y = [(a*sqft+b) for sqft in test_x]
loss = compute_loss(pred_y,test_y)
print(f'final loss: {loss}')

plt.plot(test_x, pred_y, label='converged line', color = 'green') # plot converged lines
plt.plot(test_x,test_y,'o') # plot true values/points
plt.xlabel('sqfeet')
plt.ylabel('price')
plt.legend()

plt.subplot(1,2,2)
plt.title('training vs validation loss')
plt.plot(range(epochs), train_loss_lst, label='training loss', color='red')
plt.plot((range(0, len(val_loss_lst*val_check), val_check)), val_loss_lst, label = 'validation loss', color='blue')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.show()

# save values into a .txt file
file = open('parameter_weights.txt', 'w')
file.write(f'epochs: {epochs} \na: {a} \nb: {b}')