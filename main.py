import matplotlib.pyplot as plt
import torchmetrics
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from prefect import flow, task
from prefect.testing.utilities import prefect_test_harness
from mnistdataset import MNISTDataset
from prefect.logging import get_run_logger
from model import MyModel
from tqdm import tqdm
import os
@task
def set_device():
    logger = get_run_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with prefect_test_harness():
        assert device != None
    logger.info(f"Your device is using = {device}")
    return device

@task
def load_data():
    logger = get_run_logger()
    logger.info("Loading MNIST Dataset")
    transform = transforms.Compose([transforms.ToTensor(),])
    trainset = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform)
    return trainset, testset

@task 
def create_dataloader(trainset, testset, device):
    logger = get_run_logger()
    logger.info("Creating Data Loader")
    try:
        train_dataset = MNISTDataset(trainset.data, trainset.targets, device)
        test_dataset = MNISTDataset(testset.data, testset.targets, device)

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=32)
        test_loader = DataLoader(test_dataset, shuffle=True, batch_size=32)
    except Exception as e:
        logger.error(f"Error creating data loader : {e}")
    return train_loader, test_loader

@task
def create_model():
    model = MyModel()
    return model

@task
def define_parameters(model, lr):
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return loss_fn, optimizer

@task
def train_model(model, n_epochs, optimizer, loss_fn, train_loader):
    logger = get_run_logger()
    model.train()
    for epoch in range(1, n_epochs + 1):
        print(f"\nEpoch = {epoch}")
        for X_batch, y_batch in tqdm(train_loader):

            X_batch = X_batch.unsqueeze(1)
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        logger.info(f'Finished epoch {epoch}, latest loss {loss}')

@task
def evaluate_model(model, device, test_loader):
    # Initialize precision and recall metrics
    precision_metric = torchmetrics.Precision(task="multiclass", num_classes=10).to(device)
    recall_metric = torchmetrics.Recall(task="multiclass", num_classes=10).to(device)
    accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=10).to(device)



    # Iterate over the test data
    print_ex = True
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            # Make predictions
            X_batch = X_batch.unsqueeze(1)
            outputs = model(X_batch)

            # Update accuracy metric
            _, predicted = torch.max(outputs.data, 1)
            # predicted = predicted.unsqueeze(1)

            _, y_batch = torch.max(y_batch.data, 1)
            # if print_ex:  
            #     print(f"x_batch dim = {X_batch.shape}")
            #     print(f"y_batch dim = {y_batch.shape}")
            #     print(f"predicted dim = {predicted.shape}")
            #     print(predicted)
            #     print(y_batch)
            #     print_ex = False
            

            accuracy_metric.update(predicted, y_batch)
            precision_metric.update(predicted, y_batch)
            recall_metric.update(predicted, y_batch)




    # Calculate average precision and recall
    accuracy = accuracy_metric.compute()
    precision = precision_metric.compute()
    recall = recall_metric.compute()
    print('Accuracy on test data: %f' % accuracy)
    print('Precision on test data: %f' % precision)
    print('Recall on test data: %f' % recall)

@task
def save_model(model, filename:str):
    if not os.path.exists("./saved_model"):
        os.makedirs("./saved_model")
    torch.save(model, f"./saved_model/{filename}")

@task
def load_model(filename):
    model = torch.load(f"./saved_model/{filename}", weights_only=False)
    return model


@flow(retries=3, retry_delay_seconds=3)
def mnist_workflow_ml(n_epochs:int, lr:float):
    device = set_device()
    trainset, testset = load_data()

    train_loader, test_loader = create_dataloader(trainset, testset, device)

    logger = get_run_logger()
    try:
        model = load_model("saved_2.pt")
    except Exception as e:
        logger.error(f"failed to load the model : {e}")
        logger.info(f"Creating new model")
        model = create_model()

    loss_fn, optim = define_parameters(model, lr)
    train_model(model, n_epochs, optim, loss_fn, train_loader)
    save_model(model, "saved_2.pt")
    evaluate_model(model, device, test_loader)




if __name__ == "__main__":
    mnist_workflow_ml(10, 0.001) 