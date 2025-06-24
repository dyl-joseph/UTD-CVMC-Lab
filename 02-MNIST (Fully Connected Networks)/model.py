import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

epochs = 100
lr = 1e-4
batch_size = 16
lambda_fn = 1e-5

train = pd.read_csv('02-MNIST (Fully Connected Networks)/train.csv')
validation = pd.read_csv('02-MNIST (Fully Connected Networks)/validate.csv')

def normalize(x_in):
    normalized = np.array(x_in)
    normalized = normalized / 255.0
    return normalized

train_labels = train[train.columns[:1]].values
validation_labels = validation[validation.columns[:1]].values

train_use = normalize(train[train.columns[1:]].values)
validation_use = normalize(validation[validation.columns[1:]].values)



class FNN():
    def __init__(self):
        train_loss = []
        val_loss_list = []
        

        self.w_1 = np.random.rand(20,784)
        self.bias_h = np.zeros((20,1)) #bias_h = b1
        self.w_2 = np.random.rand(10,20)
        self.bias_o = np.zeros((10,1)) #bias_o = b2
        for epoch in range(epochs):
            loss = 0
            for batch in range(0,train_use.shape[0],batch_size):
                loss+=self.backward(batch)          
            val_loss = self.validation()
            loss/=train_use.shape[0]
            val_loss/=validation_use.shape[0]
            print(loss)
            print(val_loss)
            train_loss.append(loss)
            val_loss_list.append(val_loss)
            
        plt.figure()

        plt.plot(range(epochs), train_loss, label = 'Training Loss', color='blue', linestyle='-')
        plt.plot(range(epochs), val_loss_list, label = 'Validation Loss', color='red', linestyle='-')

        plt.legend()
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.savefig('model_accuracy.png')


        plt.show()

        
            
    def forward(self,x_in):
        x_h = np.dot(self.w_1,x_in) + self.bias_h
        
        x_h = x_h.reshape((20,1))
        ReLU_x_h = self.ReLU(x_h)

        x_o = np.dot(self.w_2,ReLU_x_h) + self.bias_o
    
        return x_o,ReLU_x_h,x_h
    def ReLU(self,x_h):        
        return np.maximum(0,x_h)
    def softmax(self,x_o):    
        x_o = np.array(x_o)
        x_o -= np.max(x_o)
        exp_scores = np.exp(x_o)
    
        return exp_scores / np.sum(exp_scores)
    def one_hot(self,label):
        vector = np.zeros((10))
        vector[label] = 1        
        return vector
    def dReLU(self, input):
        return (input>0).astype(float) # returns boolean value of 0 or 1. dReLU/dX = 1 if ReLU>0 or dReLU/dX = 0  if ReLU≤0
    def CELoss(self, softmax_logits,y_true):
        small_num = 1e-15 # small number to avoid log(0) error
        softmax_logits = np.clip(softmax_logits,a_min=small_num,a_max=1-small_num)
        return -np.sum(y_true*np.log(softmax_logits))
    def L2(self, weights):
        return (np.sum(weights**2))**(1/2)
    def backward(self,start):
        CELoss = 0
        for i in range(0,batch_size):
            x_in = np.array(train_use[start+i])
            x_in = x_in.reshape((784,1))                            
            
            y_true = self.one_hot(train_labels[start+i])
            y_true = np.array(y_true)
            y_true = y_true.reshape((10,1))
            
            
            x_o,ReLU_x_h,x_h = self.forward(x_in)
            
            softmax_logits = self.softmax(x_o) # logits = x_o, softmax_logits = softmax_x_o 
            CELoss += self.CELoss(softmax_logits, y_true)
            #Layer 2
            delta_2 = softmax_logits-y_true        
            
            grad_w_2 = np.dot(delta_2,np.transpose(ReLU_x_h)) #TODO: re-check math | ReLU_x_h output used because this is what directly affects W_2, not x_h
            grad_b_2 = delta_2

            # Layer 1
            delta_1 = np.dot(np.transpose(delta_2), self.w_2) * self.dReLU(x_h)
            delta_1 = delta_1[train_labels[start+i]]
            grad_w_1 = np.dot(np.transpose(delta_1), np.transpose(x_in))
            grad_b_1 = np.transpose(delta_1)
            

        # update values
        L_2_W_2 = lambda_fn*self.L2(self.w_2)
        self.w_2 -= lr*((1/batch_size)* grad_w_2 + L_2_W_2)
        self.bias_o -= lr*((1/batch_size)*grad_b_2)

        L_2_W_1 = lambda_fn*self.L2(self.w_1)
        self.w_1 -= lr*((1/batch_size)*grad_w_1 + L_2_W_1)
        self.bias_h -= lr*((1/batch_size)*grad_b_1)

        return CELoss
    def validation(self):
        loss = 0
        for i in range(validation_use.shape[0]):
            x_in = np.array(validation_use[i])
            x_in = x_in.reshape((784,1))    
        
            y_true = self.one_hot(validation_labels[i])
            y_true = np.array(y_true)
            y_true = y_true.reshape((10,1))

            x_o,_,_ = self.forward(x_in) # if i dont include non-needed variables, code breaks
            softmax_logits = self.softmax(x_o)
            loss += self.CELoss(softmax_logits,y_true)
        return loss


FNN()