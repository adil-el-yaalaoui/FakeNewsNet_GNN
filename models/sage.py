import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool
from torch_geometric.utils import add_self_loops,degree
import torch.nn.functional as F


class SageLayer(nn.Module):
    def __init__(self, in_feature,out_feature,aggregate_fn="sum") -> None:
        super().__init__()

        self.in_feature=in_feature
        self.out_feature=out_feature
        self.agg_fn=aggregate_fn
        self.relu=nn.ReLU()
        self.proj=nn.Linear(in_features=2*in_feature, # we concatenate old vector with aggregated neighbors
                            out_features=out_feature) 

    
    def aggregate_neighbord(self,x,edge_index):
        nb_of_nodes=x.shape[0]

        neighborhood_sums = torch.zeros((nb_of_nodes,x.shape[1]), device=x.device)
        neighborhood_sums.scatter_add_(0, edge_index[1].unsqueeze(1).expand(-1, x.shape[1]), x[edge_index[0]])

        return neighborhood_sums #sum of each neighbors feature to the node 
    

    def forward(self,x,edge_index):

        aggregated_neighbors=self.aggregate_neighbord(x,edge_index)

        concat_features=torch.concat([x,aggregated_neighbors],axis=1)

        projected_features=self.relu(self.proj(concat_features))

        norm_column=F.normalize(projected_features,p=2,dim=-1)

        return norm_column
    


class SageModel(nn.Module):
    def __init__(self, feature_size,hidden_dim,n_classes) -> None:
        super().__init__()
        self.name="custom_sage"
        self.sage1=SageLayer(feature_size,hidden_dim)
        self.lin=nn.Linear(hidden_dim,n_classes)

    def forward(self,x,edge_index,batch):

        out1=self.sage1(x,edge_index)
        out2=global_max_pool(out1,batch)

        out2=self.lin(out2)

        return out2.squeeze(1)






