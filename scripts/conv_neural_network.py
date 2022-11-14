import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# https://towardsdatascience.com/logistic-regression-on-mnist-with-pytorch-b048327f8d19
# https://discuss.pytorch.org/t/how-to-load-data-from-a-csv/58315/8

class NBADataset(torch.utils.data.Dataset):
    def __init__(self, x_path, y_path):
        y_data = pd.read_csv(y_path)
        # print(y_data.head())
        x_data = pd.read_csv(x_path)
        # print(x_data.head())
        x_data.fillna(1)
        y_data.fillna(1)
        xtrain, xtest, ytrain, ytest = train_test_split(x_data, y_data, test_size=0.2)
        self.X_train = torch.tensor(x_data.values, dtype=torch.float32)
        self.y_train = torch.tensor(y_data.values, dtype=torch.float32)
        self.X_train = torch.nan_to_num(self.X_train, nan=1)
        self.y_train = torch.nan_to_num(self.y_train, nan=1)
        self.X_test = torch.tensor(x_data.values, dtype=torch.float32)
        self.y_test = torch.tensor(y_data.values, dtype=torch.float32)
        self.X_test = torch.nan_to_num(self.X_train, nan=1)
        self.y_test = torch.nan_to_num(self.y_train, nan=1)

    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()
        #self.linear = torch.nn.Linear(input_dim, output_dim)

        self.conv1 = torch.nn.Conv2d(1, 6, 5, stride=3, padding=0)

        self.ReLU = torch.nn.ReLU()

        self.max = torch.nn.MaxPool2d(2, stride=2)

        self.conv2 = torch.nn.Conv2d(6, 16, 5, stride=3, padding=0)

        self.fc0 = torch.nn.Linear(36944, 400) # this is weird definitely not correct
        self.fc1 = torch.nn.Linear(400, 120)

        self.fc2 = torch.nn.Linear(120, 84)

        self.fc3 = torch.nn.Linear(84, 10)
        self.fc4 = torch.nn.Linear(10, output_dim)

    def forward(self, x):
        # print("X shape at beginning: ", x.shape)
        x = self.conv1(x)
        x = self.ReLU(x)
        x = self.max(x)
        # print("X shape after conv1: ", x.shape)
        x = self.conv2(x)
        x = self.ReLU(x)
        x = self.max(x)
        # print("X shape after conv2: ", x.shape)
        x = torch.flatten(x, 1)
        # print("X shape after flatten: ", x.shape)
        x = self.fc0(x)
        x = self.ReLU(x)
        x = self.fc1(x)
        x = self.ReLU(x)
        x = self.fc2(x)
        x = self.ReLU(x)
        x = self.fc3(x)
        x = self.ReLU(x)
        x = self.fc4(x)
        return x

        #return outputs

model = NeuralNetwork(51, 1) # depends on our dataset dimensions, output_dim # how we want to split our features)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=.09)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
criterion = torch.nn.L1Loss()
dataset = NBADataset("../FINALIZED_DATA/fraverage_output.csv", "../FINALIZED_DATA/sorted_points.csv")
epochs = 30000
# print(torch.max(dataset.X_train))
# print(torch.min(dataset.X_train))
for epoch in range(epochs):
    y_prediction = model(dataset.X_train[None, None, ...])
    # print(y_prediction)
    loss = criterion(y_prediction, dataset.y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 100 == 0:
        print('epoch:', epoch + 1, ',loss=',loss.item())

y_prediction = model(dataset.X_test)
# print(y_prediction)
loss = criterion(y_prediction, dataset.y_test)
loss.backward()
print(loss.item())