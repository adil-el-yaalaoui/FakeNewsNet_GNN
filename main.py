import torch
import torch.nn as nn
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected
from models import GATClassifier,GNNClassifier, GenericModel,GCNModel,SageModel

from utils import train_loop
import os

if not os.path.exists("Results"):
    os.makedirs("Results")

dataset_name="gossipcop"
feature_name="bert"



if dataset_name=="politifact":
    batch_size=32
else:
    batch_size=128

train_dataset = UPFD("", dataset_name, feature_name, 'train', ToUndirected())
val_dataset = UPFD("", dataset_name, feature_name, 'val', ToUndirected())
test_dataset = UPFD("", dataset_name, feature_name, 'test', ToUndirected())

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)


if feature_name=="spacy":
    feature_size=300

elif feature_name=="bert":
    feature_size=768

elif feature_name=="profile":
    feature_size=10

elif feature_name=="content":
    feature_size=310

hidden_dim=512
### List of models ### 
    
custom_gat=GATClassifier(feature_size,hidden_dim) 
real_gat=GNNClassifier(feature_size,hidden_dim)

custom_gcn=GCNModel(in_channels=feature_size,hidden_dim=hidden_dim,num_class=1)
real_gcn=GenericModel(num_features=feature_size,hidden_dim=hidden_dim,num_classes=1,model="gcn",concat=True)

custom_sage=SageModel(feature_size=feature_size,hidden_dim=hidden_dim,n_classes=1)
real_sage = GenericModel(num_features=feature_size,hidden_dim=hidden_dim,num_classes=1,model="sage",concat=True)


### Training parameters ###
criterion=nn.BCEWithLogitsLoss()
epochs=30

device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(device)

if __name__=="__main__":
    new_gat,losses,scores,=train_loop(model=custom_gat,
                                  loss_fn=criterion,
                                  train_loader=train_loader,
                                  val_loader=val_loader,
                                  test_loader=test_loader,
                                  max_epochs=epochs,
                                  device=device,
                                  dataset_name=dataset_name,
                                  feature_name=feature_name,
                                  lr=0.0005)