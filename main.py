import torch
import torch.nn as nn
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected
import numpy as np
from sklearn.metrics import f1_score
from models import GATClassifier,GNNClassifier, GenericModel,GCNModel,SageModel

from utils import train_loop, test_loop
import os

if not os.path.exists("Results"):
    os.makedirs("Results")

dataset_name="politifact"
feature_name="profile"


train_dataset = UPFD("", dataset_name, feature_name, 'train', ToUndirected())
val_dataset = UPFD("", dataset_name, feature_name, 'val', ToUndirected())
test_dataset = UPFD("", dataset_name, feature_name, 'test', ToUndirected())

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)


if feature_name=="spacy":
    feature_size=300

elif feature_name=="bert":
    feature_size=768

elif feature_name=="profile":
    feature_size=10


gat_perso=GATClassifier(feature_size,1024) 
classif=GNNClassifier(feature_size,1024)

gcn_model=GCNModel(in_channels=feature_size,hidden_dim=1024,num_class=1)

sage=SageModel(feature_size=feature_size,hidden_dim=60,n_classes=1)

generic=GenericModel(num_features=feature_size,hidden_dim=1024,num_classes=1,model="gcn",concat=True)
generic_sage = GenericModel(num_features=feature_size,hidden_dim=1024,num_classes=1,model="sage",concat=True)

criterion=nn.BCEWithLogitsLoss()
epochs=50

device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)
if __name__=="__main__":
    new_gat,losses,scores,=train_loop(model=gat_perso,
                                  loss_fn=criterion,
                                  train_loader=train_loader,
                                  val_loader=val_loader,
                                  test_loader=test_loader,
                                  max_epochs=epochs,
                                  device=device,
                                  dataset_name=dataset_name,
                                  feature_name=feature_name,
                                  lr=0.05)