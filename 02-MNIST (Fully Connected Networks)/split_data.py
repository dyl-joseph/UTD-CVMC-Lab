import pandas as pd
import numpy as np


df = pd.read_csv('02-MNIST (Fully Connected Networks)/train_to_modify.csv')
slice_index = int(df.shape[0]*0.8)

train = df[:slice_index] # 80%
validation = df[slice_index+1:] # 20% of data used to validate the training set

train.to_csv('02-MNIST (Fully Connected Networks)/train.csv', index = False)
validation.to_csv('02-MNIST (Fully Connected Networks)/validate.csv', index=False)