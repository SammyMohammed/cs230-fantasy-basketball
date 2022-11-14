import torch
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

# https://towardsdatascience.com/logistic-regression-on-mnist-with-pytorch-b048327f8d19
# https://discuss.pytorch.org/t/how-to-load-data-from-a-csv/58315/8

class NBADataset(torch.utils.data.Dataset):
    def __init__(self, x_path, y_path):
        y_data = pd.read_csv(y_path)
        print(y_data.head())
        x_data = pd.read_csv(x_path)
        print(x_data.head())
        x_data.fillna(1)
        y_data.fillna(1)
        xtrain, xtest, ytrain, ytest = train_test_split(x_data, y_data, test_size=0.2)
        self.X_train = torch.tensor(xtrain.values, dtype=torch.float32)
        self.y_train = torch.tensor(ytrain.values, dtype=torch.float32)
        self.X_train = torch.nan_to_num(self.X_train, nan=1)
        self.y_train = torch.nan_to_num(self.y_train, nan=1)
        self.X_test = torch.tensor(xtest.values, dtype=torch.float32)
        self.y_test = torch.tensor(ytest.values, dtype=torch.float32)
        self.X_test = torch.nan_to_num(self.X_test, nan=1)
        self.y_test = torch.nan_to_num(self.y_test, nan=1)

    def __len__(self):
        return len(self.y_train)
    
    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = self.linear(x)
        return outputs

model = LogisticRegression(51, 1) 
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
criterion = torch.nn.L1Loss()
dataset = NBADataset("../FINALIZED_DATA/weighted_average_output.csv", "../FINALIZED_DATA/points.csv")
epochs = 10000
for epoch in range(epochs):
    y_prediction = model(dataset.X_train)
    loss = criterion(y_prediction, dataset.y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 100 == 0:
        print('epoch:', epoch + 1, ',loss=',loss.item())

y_prediction = model(dataset.X_test)
loss = criterion(y_prediction, dataset.y_test)
loss.backward()
print(loss.item())