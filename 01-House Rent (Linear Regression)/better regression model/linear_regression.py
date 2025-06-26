import argparse, csv, random, copy
# organize how to get inputs for script
import matplotlib.pyplot as plt

def parser():

    args_reader = argparse.ArgumentParser(description="Script to train a linear regression model with two parameteres on house rent datasest.")

    args_reader.add_argument('--csv_file_path', type=str, default='/Users/dylanjoseph/Library/CloudStorage/OneDrive-Personal/Code/CVMC Lab Work/01-House Rent (Linear Regression)/housing_train.csv', help = 'Path to the csv file containing house rent data')
    
    # initialize hyper-parameters
    args_reader.add_argument('--lr', type=float, default=1e-2, help='learning rate for training the linear regression model')
    args_reader.add_argument('--epochs', type=int, default=300, help='number of epochs to train the model on.')
    args_reader.add_argument('--val_interval', type=int, default=25, help='number of updates between each validation') 
    args_reader.add_argument('--lr_drop_epoch', type=int, default=200, help='Epoch at which lr drops.')
    args_reader.add_argument('--lr_drop_factor', type=float, default=0.1, help='Factor by which the learning rate drops.' )

    args = args_reader.parse_args()

    return args

def compute_losses(pred, true_val):
    total_loss = 0
    N = len(pred)
    for ind in range(N):
        total_loss+= (1/N) * ((pred[ind]-true_val[ind])**2)
    return total_loss

def compute_grad_a(train_sqfeet, pred_price, train_prices):
    grad_a = 0
    N = len(train_sqfeet)
    for i in range(N):
        grad_a += (1/N) * (pred_price[i]-train_prices[i]) * 2 * train_sqfeet[i]

    return grad_a


def compute_grad_b(pred_price, train_prices):
    grad_b = 0
    N = len(train_prices)
    for i in range(N):
        grad_b += (1/N) * (pred_price[i]-train_prices[i]) * 2

    return grad_b

if __name__ == "__main__":

    args = parser()
    print('Training with learning rate: ', args.lr)
    print('Epochs: ', args.epochs)

    print('Reading data from file: ', args.csv_file_path)
    file_handler = open(args.csv_file_path)
    csv_reader = csv.reader(file_handler,) # points to every line in the file
    all_lines = list(csv_reader)
    file_handler.close()
    
    all_lines = all_lines[1:]
    all_sqfeet = [float(x[6]) for x in all_lines] # string --> float
    all_prices = [float(x[4]) for x in all_lines]


# split data into train, validation, and test splits
total_N = len(all_sqfeet)
train_N = int(0.8*total_N)
val_N = int(0.1*total_N)
test_N = total_N - (train_N+val_N)

train_sqfeet = all_sqfeet[0:train_N]
train_prices = all_prices[0:train_N]

max_sqfeet = max(train_sqfeet)
max_price = max(train_prices)

train_sqfeet = [(val) / (max_sqfeet) for val in train_sqfeet] # Normalize [-1,1]
train_prices = [(val) / (max_price) for val in train_prices] # Normalize [-1,1]




val_sqfeet = all_sqfeet[train_N:train_N+val_N]
val_prices = all_prices[train_N:train_N+val_N]

val_sqfeet = [(val) / (max_sqfeet) for val in val_sqfeet] # Normalize [-1,1]
val_prices = [(val) / (max_price) for val in val_prices] # Normalize [-1,1]





test_sqfeet = all_sqfeet[train_N+val_N:]
test_prices = all_prices[train_N+val_N:]

test_sqfeet = [(val) / (max_sqfeet) for val in test_sqfeet] # Normalize [-1,1]
test_prices = [(val) / (max_price) for val in test_prices] # Normalize [-1,1]


    
a = random.random()
b = random.random()

a_lst = []
b_lst = []
val_loss_lst = []
train_loss_lst = []

min_val_loss = 10000
min_val_loss_index = -1
learning_rate_updated = False


pred_price_test = [a*sqft+b for sqft in test_sqfeet]
plt.figure()
plt.subplot(1,2,1)
plt.title('line of best fit on test set')
plt.plot(test_sqfeet, pred_price_test, label= 'initial', color='red') # regression line
plt.plot(test_sqfeet,test_prices, 'bo') # true values
plt.xlabel('sq feet')
plt.ylabel('price')


print(f'Initial values a: {a}, b: {b}')


for epoch in range(args.epochs):
    if (epoch % args.val_interval == 0):
        a_lst.append(copy.deepcopy(a))
        b_lst.append(copy.deepcopy(b))
        val_pred_price = [a*val_sq_ft+b for val_sq_ft in val_sqfeet]
        val_loss = compute_losses(val_pred_price, val_prices)
        val_loss_lst.append(copy.deepcopy(val_loss))
        print(" ")
        print('val_loss:', val_loss)
        if (val_loss < min_val_loss):
            min_val_loss = val_loss
            min_val_loss_index = int(epoch/args.val_interval)
    if (epoch > args.lr_drop_epoch and not learning_rate_updated):
        args.lr *= args.lr_drop_factor
        learning_rate_updated = True
    
    pred_price = [a*sqft + b for sqft in train_sqfeet] # Forward pass
    loss = compute_losses(pred_price, train_prices)
    train_loss_lst.append(copy.deepcopy(loss))
    # print('loss: ', loss)
    
    
    # Backprop
    grad_a = compute_grad_a(train_sqfeet, pred_price, train_prices)
    grad_b = compute_grad_b(pred_price, train_prices)
    
    
    a -= args.lr*(grad_a)
    b -= args.lr*(grad_b)

    
a_test = a_lst[min_val_loss_index]
b_test = b_lst[min_val_loss_index]

pred_price_test = [(a_test * test_sqfeet[i]) + b_test for i in range(len(test_sqfeet))]

print('final loss: ', compute_losses(pred_price_test, test_prices))



plt.plot(test_sqfeet, pred_price_test, label= 'line of best fit', color='green')
plt.legend()

plt.subplot(1,2,2)
plt.title('train vs validation loss')
plt.plot(range(len(train_loss_lst)), train_loss_lst, label='train loss', color = 'red')
plt.plot((range(0, len(val_loss_lst*args.val_interval), args.val_interval)), val_loss_lst, label='validation loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()



plt.savefig('model_accuracy.png')
plt.show()