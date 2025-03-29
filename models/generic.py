import torch
import torch.nn as nn
from torch_geometric.nn import global_max_pool, GCNConv, SAGEConv, SimpleConv

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

        output= global_max_pool(output,batch)

        if self.concat:
            # We need to create a tensor of shape (batch_size, num_nodes, features)
            # to make it pass through the linear projection layer

            number_of_graphs=torch.max(batch)+1
            
            skip=torch.stack([x[(batch==idx).nonzero().squeeze()[0]] for idx in range(number_of_graphs) ]) # tensor of shape (batch_size, features) because we took only one node

            skip=self.relu(self.lin0(skip)) 

            output=torch.cat([output,skip],dim=1)

            output=self.relu(self.lin1(output))
        
        return self.lin2(output).squeeze(1)

