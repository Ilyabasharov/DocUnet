import torch, os
from torch import nn
from tqdm import tqdm
from model import Doc_UNet
from dataset import Dataset

class Loss(nn.Module):
    def __init__(self, coef: float=2.):
        super().__init__()
        self.coef = coef
        
    def forward(self, gt, pred):
        ind_gt = gt < 0
        ind_pred = pred > 0
        
        l1 = (pred[ind_gt & ind_pred] + 1).sum()
        
        ind_gt = ~ind_gt
        l2 = torch.abs(pred[ind_gt] - gt[ind_gt]).sum()
        
        return l2 + self.coef*l1

def collate_fn(batch):

    images = list()
    masks = list()

    for b in batch:
        images.append(b[0])
        masks.append(b[1])

    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)

    return images, masks
    
def train(model, optimizer, scheduler, dataloaders_dict, epochs, device):
    model.to(device)
    loss = Loss()
    
    for epoch in range(epochs):
        for phase in dataloaders_dict:
            
            model.train() if phase == 'train' else model.eval()
            loss_per_epoch = 0
            
            for images, masks in tqdm(dataloaders_dict[phase], desc = phase):
                
                images = images.to(device)
                gt = masks.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    pred = model(images)
                    error = loss(gt, pred)
                    
                    if phase == 'train':
                        error.backward()
                        optimizer.step()
                        
                    loss_per_epoch += error
                    
            print('Epoch:', epoch, 'Phase:', phase, 'Loss:', loss_per_epoch.mean())
        
        scheduler.step()
        if phase == 'train':
            torch.save(model.state_dict(), f'models/{epoch}.pth')
            
def main():
    os.makedirs('models', exist_ok=True)
    
    model = Doc_UNet(3, 2)
    device = torch.device("cuda:1" if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], 
        lr=1e-3, weight_decay = 1e-3, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    dataset = Dataset('dataset/0')
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [int(len(dataset)*0.8), len(dataset) - int(len(dataset)*0.8)]
    )
    
    dataset = {'train': train_dataset, 'val': val_dataset}
    dataloader = {
        phase: torch.utils.data.DataLoader(
            dataset[phase], 
            batch_size = 1, 
            shuffle = True, 
            drop_last = True,
            collate_fn=collate_fn,
            num_workers=1,
            pin_memory=False,
        )
        
        for phase in dataset}
    
    train(model, optimizer, scheduler, dataloader, 10, device)
    
if __name__ == '__main__':
    main()
