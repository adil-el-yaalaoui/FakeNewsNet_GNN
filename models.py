import torch
import numpy as np
import matplotlib.pyplot as plt

import torch.nn as nn
import torch_geometric.nn as graphnn
from sklearn.metrics import f1_score
from torch_geometric.datasets import PPI
from torch_geometric.loader import DataLoader

from torch_geometric.nn import GATConv, global_max_pool, GCNConv, SAGEConv, SimpleConv


class GATLayer(nn.Module):
    def __init__(self,nb_heads,in_feature_size,out_feature_size,activation,concat) :
        super().__init__()
        self.nb_heads=nb_heads
        self.in_feature_size=in_feature_size
        self.out_feature_size=out_feature_size

        self.linear_proj = nn.Linear(in_feature_size, nb_heads * out_feature_size, bias=False)

        self.scoring_fn_target = nn.Parameter(torch.Tensor(1, nb_heads, out_feature_size))
        self.scoring_fn_source = nn.Parameter(torch.Tensor(1, nb_heads, out_feature_size))

        self.leakyReLU = nn.LeakyReLU(0.2)  
        self.softmax = nn.Softmax(dim=-1)  
        self.activation = activation
        self.concat=concat
        self.skip_proj = nn.Linear(in_feature_size, nb_heads * out_feature_size, bias=False)

        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.linear_proj.weight)
        nn.init.xavier_uniform_(self.scoring_fn_target)
        nn.init.xavier_uniform_(self.scoring_fn_source)

    def lift(self,score_source,score_target,projected_nodes,edge_index):
        source_index=edge_index[0]
        targ_index=edge_index[1]

        score_source_lifted=score_source.index_select(0,source_index)
        score_target_lifted=score_target.index_select(0,targ_index)

        projected_nodes_lifted=projected_nodes.index_select(0,source_index)

        return score_source_lifted,score_target_lifted,projected_nodes_lifted

    def softmax_denominator(self,exp_score_per_edge,edge_index,nb_of_nodes):
        """
        This function computes fo each nodes the exponential sum of the neighbor nodes
        """
        index_modified=edge_index.unsqueeze(-1).expand_as(exp_score_per_edge)

        neighborhood_sums = torch.zeros((nb_of_nodes,exp_score_per_edge.shape[1]), device=exp_score_per_edge.device)

        neighborhood_sums.scatter_add_(0, index_modified, exp_score_per_edge)

        return neighborhood_sums.index_select(0,edge_index)
    
    def softmax_neighborhood(self,score_per_edge,edge_index,nb_of_nodes):
        """
        This function computes the exponential of the edge coefficient
        Then applies the local softmax
        """

        exp_score_per_edge=(score_per_edge-score_per_edge.max()).exp() #numerical stability
        denominator=self.softmax_denominator(exp_score_per_edge,edge_index,nb_of_nodes)
        attentions_per_edge = exp_score_per_edge / (denominator + 1e-16)

        return attentions_per_edge.unsqueeze(-1)


    def aggregate_neighbors(self,projected_nodes_attention_weighted,edge_index,x):


        out_nodes_features = torch.zeros((x.shape[0],
                                          projected_nodes_attention_weighted.shape[1],
                                          projected_nodes_attention_weighted.shape[2]), device=x.device)
        
        index_modified=edge_index.unsqueeze(-1).unsqueeze(-1).expand_as(projected_nodes_attention_weighted)

        out_nodes_features.scatter_add_(0, index_modified, projected_nodes_attention_weighted)

        return out_nodes_features
    
    def skip_connection(self,in_features,out_features):

        if out_features.shape[-1] == in_features.shape[-1]:  # if FIN == FOUT
                # unsqueeze does this: (N, FIN) -> (N, 1, FIN), out features are (N, NH, FOUT) so 1 gets broadcast to NH
                # thus we're basically copying input vectors NH times and adding to processed vectors
                out_features += in_features.unsqueeze(1)
        else:
                # FIN != FOUT so we need to project input feature vectors into dimension that can be added to output
                # feature vectors. skip_proj adds lots of additional capacity which may cause overfitting.
                out_features += self.skip_proj(in_features).view(-1, self.nb_heads, self.out_feature_size)
            
        
        if self.concat:
            # shape = (N, NH, FOUT) -> (N, NH*FOUT)
            out_features = out_features.view(-1, self.nb_heads * self.out_feature_size)
        else:
            # shape = (N, NH, FOUT) -> (N, FOUT)
            out_features = out_features.mean(dim=1)

        return out_features

    def forward(self,x,edge_index):

        nb_of_nodes=x.shape[0]

        projected_nodes=self.linear_proj(x).view(-1,self.nb_heads,self.out_feature_size) #of shape (N,H,O)

        scores_source = (projected_nodes * self.scoring_fn_source).sum(dim=-1)
        scores_target = (projected_nodes * self.scoring_fn_target).sum(dim=-1)
        score_source_lifted,score_target_lifted,projected_nodes_lifted=self.lift(scores_source,scores_target,projected_nodes,edge_index)
    
        score_per_edge=self.leakyReLU(score_source_lifted+score_target_lifted)
        
        # Now we have the post attention mechanism for source nodes and target nodes
        # We now apply the softmax only to the neighbors

        attentions_per_edge = self.softmax_neighborhood(score_per_edge, edge_index[1], nb_of_nodes)

        projected_nodes_attention_weighted = projected_nodes_lifted * attentions_per_edge
        # Final step is to aggregate the same weighted (by attention coeff) nodes with their neighbors

        updated_nodes_features=self.aggregate_neighbors(projected_nodes_attention_weighted,edge_index[1],x)


        updated_nodes_features=self.skip_connection(x,updated_nodes_features)
        
        #updated_nodes_features=torch.mean(updated_nodes_features,dim=1)
        return self.activation(updated_nodes_features) if self.activation!=None else updated_nodes_features



class GAT_Model(nn.Module):
    def __init__(self, head,final_head,in_feature_size,out_feature_size,nb_classes):
        super().__init__()
        self.layer1=GATLayer(nb_heads=head,in_feature_size=in_feature_size,out_feature_size=out_feature_size,activation=nn.ELU(),concat=True)

        self.layer2=GATLayer(nb_heads=head,in_feature_size=head*out_feature_size,out_feature_size=out_feature_size,activation=nn.ELU(),concat=True)

        self.last_layer=GATLayer(nb_heads=final_head,in_feature_size=head*out_feature_size,out_feature_size=nb_classes,activation=None,concat=False)

    def forward(self,x,edge_index):
        out1=self.layer1(x,edge_index)

        out2=self.layer2(out1,edge_index)

        out3=self.last_layer(out2,edge_index)

        return out3


class GNNClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels,heads=4,concat=False)
        self.conv2 = GATConv(hidden_channels, hidden_channels,heads=6,concat=False)

        self.lin = torch.nn.Linear(hidden_channels, 1)
        self.activation=nn.ReLU()
        self.name="gat"

    def forward(self, x, edge_index,batch):
        x = self.conv1(x, edge_index)
        x=self.activation(x)
        x = self.conv2(x, edge_index)
        x=self.activation(x)

        x = global_max_pool(x,batch)  # Agrégation du graphe
        x = self.lin(x)
        return x.squeeze(1)

class GATClassifier(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.gat=GAT_Model(head=4,final_head=6,in_feature_size=in_channels,out_feature_size=hidden_channels,nb_classes=1)
        self.name="gat_perso"

    def forward(self,x,edge_index,batch):
        x=self.gat(x,edge_index)

        x=global_max_pool(x,batch)
        return x.squeeze(1)


class GenericModel(nn.Module):
    def __init__(self, num_features,hidden_dim,num_classes,model,concat=False) -> None:
        """
        Generic class to test classic models for our dataset
        """
        super().__init__()
        self.num_features=num_features
        self.hidden_dim=hidden_dim
        self.num_classes=num_classes
        self.model=model
        self.concat=concat
        self.relu=nn.ReLU()
        

        if self.model=="gcn":
            # Implement a graph convolution network
            self.conv=GCNConv(self.num_features,self.hidden_dim)
            self.name="gcn"
        
        elif self.model=="sage":
            # Implement a sage graph convolutional network
            self.conv=SAGEConv(self.num_features,self.hidden_dim)
            self.name="sage"
        
        else:
            # Default model which is a simple message passing with average neighbors nodes features
            self.conv=SimpleConv(aggr="mean")
            self.concat=False 
            self.name="simple"

        if self.concat:
            # This is used to project the original input and concatenante it with th output of the model
            # Can be seen as a type of skip connection
            self.lin0 = nn.Linear(self.num_features, self.hidden_dim)
            self.lin1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        
        self.lin2=nn.Linear(self.hidden_dim,self.num_classes)

    
    def forward(self,x,edge_index,batch):

        output=self.relu(self.conv(x,edge_index))

        output= global_max_pool (output,batch)

        if self.concat:
            # We need to create a tensor of shape (batch_size, num_nodes, features)
            # to make it pass through the linear projection layer

            number_of_graphs=torch.max(batch)+1
            
            skip=torch.stack([x[(batch==idx).nonzero().squeeze()[0]] for idx in range(number_of_graphs) ]) # tensor of shape (batch_size, features) because we took only one node

            skip=self.relu(self.lin0(skip)) 

            output=torch.cat([output,skip],dim=1)

            output=self.relu(self.lin1(output))
        
        return self.lin2(output).squeeze(1)

