import torch
import torch.nn.functional as F
from torch.nn import Linear
import torch.nn as nn
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import ToUndirected
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_max_pool
import gc
import numpy as np
from sklearn.metrics import f1_score

import logging





train_dataset = UPFD("", "gossipcop", "spacy", 'train', ToUndirected())
val_dataset = UPFD("", "gossipcop", "spacy", 'val', ToUndirected())
test_dataset = UPFD("", "gossipcop", "spacy", 'test', ToUndirected())

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
#data2=UPFD("",name="gossipcop",feature="profile")

logging.basicConfig(
    filename='GAT_Gossipcop.log',   # Nom du fichier de log
    level=logging.INFO,    # Niveau du log (INFO, DEBUG, WARNING, ERROR, CRITICAL)
    format='%(asctime)s -- %(message)s'  # Format du message
)

class GNNClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels,heads=4,concat=False)
        self.conv2 = GATConv(hidden_channels, hidden_channels,heads=6,concat=False)

        self.lin = torch.nn.Linear(hidden_channels, 1)
        self.activation=nn.ReLU()

    def forward(self, x, edge_index,batch):
        x = self.conv1(x, edge_index)
        x=self.activation(x)
        x = self.conv2(x, edge_index)
        x=self.activation(x)

        x = global_max_pool(x,batch)  # Agrégation du graphe
        x = self.lin(x)
        return x.squeeze(1)
    
device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

classif=GNNClassifier(300,1024).to(device)

criterion=nn.BCEWithLogitsLoss()
other_loss=nn.BCELoss()
optim=torch.optim.Adam(classif.parameters(),lr=0.005)
epochs=20

for j in range(epochs):
    losses=[]
    f1_scores_list=[]
    test_scores=[]
    for i,batch in enumerate(train_loader):
        batch=batch.to(device)
        optim.zero_grad()
        batch.y=batch.y.to(torch.float32)
        
        out=classif(batch.x,batch.edge_index,batch.batch)
        loss=criterion(out,batch.y)


        loss.backward()
        optim.step()
        losses.append(loss.item())

    for k,val_batch in enumerate(val_loader):
        classif.eval()
        val_batch=val_batch.to(device)
        out_val=classif(val_batch.x,val_batch.edge_index,val_batch.batch)
        predict = np.where(out_val.detach().cpu().numpy() >= 0, 1, 0)
        f1_scores_list.append(f1_score(val_batch.y.float().cpu().numpy(),predict,average="micro"))

    for n,test_batch in enumerate(test_loader):
        test_batch.y=test_batch.y.to(torch.float32)
        test_batch=test_batch.to(device)
        out_test=classif(test_batch.x,test_batch.edge_index,test_batch.batch)
        predict_test = np.where(out_test.detach().cpu().numpy() >= 0, 1, 0)
        test_scores.append(f1_score(test_batch.y.float().cpu().numpy(),predict_test,average="micro"))
    logging.info(f"Epoch : {j+1} Loss : {np.mean(losses)} F1-score : {np.mean(f1_scores_list)} F1-score test : {np.mean(test_scores)}")
    print("Epoch : ",j+1 ," Loss : ",np.mean(losses), " F1-score : ",np.mean(f1_scores_list), " F1-score test : ",np.mean(test_scores))

torch.save(classif.state_dict(),"GAT_Gossipcop_spacy.pth")