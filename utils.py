import torch
from sklearn.metrics import f1_score
import numpy as np
import logging



def train_loop(model,loss_fn,optimizer,train_loader,val_loader,max_epochs,device,dataset_name,feature_name):

    file_name="GNN_"+dataset_name+"_"+feature_name+".log"

    logging.basicConfig(
    filename=file_name,   
    level=logging.INFO,    
    format='%(asctime)s -- %(message)s'  
    )

    
    model.to(device)
    full_loss_list=[]
    full_score_list=[]
    for j in range(max_epochs):
        losses=[]
        f1_scores_list=[]
        for i,batch in enumerate(train_loader):
            batch=batch.to(device)
            optimizer.zero_grad()
            batch.y=batch.y.to(torch.float32)
            
            out=model(batch.x,batch.edge_index,batch.batch)
            loss=loss_fn(out,batch.y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        with torch.no_grad():
            model.eval()
            for k,val_batch in enumerate(val_loader):
                val_batch=val_batch.to(device)
                out_val=model(val_batch.x,val_batch.edge_index,val_batch.batch)
                predict = np.where(out_val.detach().cpu().numpy() >= 0, 1, 0)
                f1_scores_list.append(f1_score(val_batch.y.float().cpu().numpy(),predict,average="micro"))

        logging.info(f"Epoch : {j+1} Loss : {np.mean(losses)} F1-score eval : {np.mean(f1_scores_list)}")
        full_loss_list.append(np.mean(losses))
        full_score_list.append(np.mean(f1_scores_list))

    return model,full_loss_list,full_score_list

def test_loop(model,test_loader,device):
    model.to(device)
    test_scores=[]
    with torch.no_grad():
        model.eval()
        for n,test_batch in enumerate(test_loader):
            test_batch.to(device)
            test_batch.y=test_batch.y.to(torch.float32)
            out_test=model(test_batch.x,test_batch.edge_index,test_batch.batch)
            predict_test = np.where(out_test.detach().cpu().numpy() >= 0, 1, 0)
            test_scores.append(f1_score(test_batch.y.float().cpu().numpy(),predict_test,average="micro"))
        
    return np.mean(test_scores)

