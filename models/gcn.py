import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool, GCNConv, SAGEConv, SimpleConv
from torch_geometric.utils import add_self_loops,degree

device = torch.device("mps") if torch.mps.is_available() else torch.device("cpu")

class GCNLayer(nn.Module):
    def __init__(self, in_size,out_size) -> None:
        super().__init__()

        self.in_size=in_size
        self.out_size=out_size
        self.relu=nn.ReLU()

        self.proj=nn.Linear(in_size,out_size)


    def transform_adj_matrix(self,edge_index,num_nodes):
        """ 
        Takes as an input the edge_index vector and returns the adjency matrix of the graph
        """
        edge_index_self_loops,_=add_self_loops(edge_index)
        values = torch.ones(edge_index.shape[1]) 
        adjacency_sparse = torch.sparse_coo_tensor(edge_index, values, (num_nodes, num_nodes),device=device)

        return adjacency_sparse


    def forward(self,x,edge_index):
        num_nodes=x.shape[0]
        adjacency_matrix=self.transform_adj_matrix(edge_index,num_nodes)

        degree_mat=torch.diag(torch.pow(degree(edge_index[0]),exponent=-0.5))


        degree_forward=torch.sparse.mm(degree_mat,adjacency_matrix)

        degree_forward=torch.sparse.mm(degree_forward,degree_mat)

        node_level_forwrd=degree_forward@x

        return self.relu(self.proj(node_level_forwrd))



class GCNModel(nn.Module):
    def __init__(self, in_channels,hidden_dim,num_class) -> None:
        super().__init__()

        self.name="gcn_perso"

        self.layer1=GCNLayer(in_size=in_channels,out_size=hidden_dim)

        #self.layer2=GCNLayer(in_size=hidden_dim,out_size=num_class)

        self.lin = torch.nn.Linear(hidden_dim, 1)

    
    def forward(self,x,edge_index,batch):

        out1=self.layer1(x,edge_index)

        out2=global_max_pool(out1,batch)

        final_out=self.lin(out2
                           )
        return final_out.squeeze(1)
