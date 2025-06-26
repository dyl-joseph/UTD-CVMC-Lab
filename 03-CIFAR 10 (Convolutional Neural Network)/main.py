import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

batch_size = 32
lr = 1e-3
epochs = 50


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = torchvision.datasets.CIFAR10(root="03-CIFAR 10 (Convolutional Neural Network)", train=True, transform=transform, download=True)
train, val = torch.utils.data.random_split(train_data, [45000, 5000])
test = torchvision.datasets.CIFAR10(root="03-CIFAR 10 (Convolutional Neural Network)", train=False, transform=transform, download=True)

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")


#TODO: write a custom dataloader
# class Custom_DataLoader(DataLoader):
#     def __init__(self, dataset, batch_size : int, shuffle : bool):
#         if (shuffle==True):
#             data = random.shuffle(dataset)

#         dataset = iter(dataset)
#         print(torch.type(dataset))
#     # def __getitem__(index):

#     # def __len__(self):
#     #     return self.len
# Custom_DataLoader(train,batch_size,True)

train_set = DataLoader(train, batch_size=batch_size, shuffle=True)
val_set = DataLoader(val, batch_size=batch_size, shuffle=True)
test_set = DataLoader(test, batch_size=batch_size, shuffle=True)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=16, momentum=0.2),
            nn.Dropout(0.1),
            nn.ReLU(),

            nn.Conv2d(in_channels = 16, out_channels = 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=32, momentum=0.2),
            nn.Dropout(0.1),
            nn.ReLU(),

            nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=64, momentum=0.2),
            nn.Dropout(0.1),
            nn.ReLU(),
        

            nn.Flatten(),

            nn.Linear(in_features=1024, out_features = 512),
            nn.BatchNorm1d(num_features=512, momentum=0.2),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(num_features=256, momentum=0.2),
            nn.Dropout(0.2),
            nn.ReLU(),

            nn.Linear(in_features=256, out_features = 10)
        )
    def forward(self,x):
        logits = self.model(x)
        
        return logits

C = CNN().to(device)




def test_params():
    num_params = 0
    for x in C.parameters():
        num_params += len(torch.flatten(x))

    print(f'{num_params:,} parameters')

    test_img, _ = next(iter(test_set))
    test_img = test_img.to(device)
    print(C(test_img).shape)

# test_params()

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(C.parameters(), lr = lr, weight_decay=1e-3)

class train():
    def __init__(self):
        for epoch in range(0,epochs):
            self.train_epoch(epoch)
            self.validate_epoch(epoch)
        self.test()
            
        
        # plt.figure()

        # plt.plot(range(epochs), train_losses, label = 'Training Loss', color='blue', linestyle='-')
        # plt.plot(range(epochs), val_losses, label = 'Validation Loss', color='red', linestyle='-')

        # plt.legend()
        # plt.xlabel("Epoch")
        # plt.ylabel("Loss")
        # plt.title("Training vs Validation Loss")


        # plt.show()
        torch.save(C.state_dict(), "03-CIFAR 10 (Convolutional Neural Network)/model_4.pt")
        print("Saved PyTorch Model State to model.pt")         

    def train_epoch(self, epoch):
        batches_loss = 0
        C.train(True)
        print('---train---')
        
        for batch_i, (train_imgs, train_labels) in enumerate(train_set):
            train_imgs, train_labels = train_imgs.to(device), train_labels.to(device)
            
            # Calc functions
            y_hat = C(train_imgs)
            loss = loss_fn(y_hat, train_labels)
            batches_loss += loss

            # Backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
            if (batch_i % 250 == 249):
                print(f'[{epoch+1},{batch_i+1}] Average Loss: {batches_loss/(250)}')
                batches_loss = 0

    def validate_epoch(self, epoch):
        C.eval()
        total_loss = 0
        print('---validate---')
        
        with torch.no_grad():
            for batch_i, (val_imgs, val_labels) in enumerate(val_set):
                val_imgs, val_labels = val_imgs.to(device), val_labels.to(device)
                y_hat = C(val_imgs)
                loss = loss_fn(y_hat, val_labels)
                total_loss += loss
                
                if (batch_i % 50 == 49):
                    print(f'[{epoch+1}, {batch_i+1}] Average Loss: {total_loss/(50)}')
                    total_loss = 0

    def test(self):
        C.eval()
        correct = 0
        total = 0

        print('---test---')
        with torch.no_grad():
            for batch_i, (test_imgs, test_labels) in enumerate(test_set):
                test_imgs, test_labels = test_imgs.to(device), test_labels.to(device)

                y_hat = C(test_imgs)

                _, predicted = torch.max(y_hat, 1)
                total+=test_labels.size(0)
                correct += (predicted==test_labels).sum().item()
        print(f'Accuracy {100 * (correct / total)}%')

train()