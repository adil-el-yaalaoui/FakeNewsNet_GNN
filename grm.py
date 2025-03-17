import torch
import torch.nn as nn
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected
import numpy as np
from sklearn.metrics import f1_score
from models import GATClassifier,GNNClassifier
from utils import train_loop, test_loop

dataset_name="gossipcop"
feature_name="spacy"


train_dataset = UPFD("", dataset_name, feature_name, 'train', ToUndirected())
val_dataset = UPFD("", dataset_name, feature_name, 'val', ToUndirected())
test_dataset = UPFD("", dataset_name, feature_name, 'test', ToUndirected())

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)




gat_perso=GATClassifier(300,1024) # 300 is the size of the features spacy
classif=GNNClassifier(300,1024)

criterion=nn.BCEWithLogitsLoss()
optim=torch.optim.Adam(gat_perso.parameters(),lr=0.05)
epochs=10

device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

new_gat,losses,scores,=train_loop(model=gat_perso,
                                  loss_fn=criterion,
                                  optimizer=optim,
                                  train_loader=train_loader,
                                  val_loader=val_loader,
                                  max_epochs=epochs,
                                  device=device,
                                  dataset_name=dataset_name,
                                  feature_name=feature_name)