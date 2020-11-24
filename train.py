import torch, os
from torch import nn
from tqdm import tqdm
from model import Doc_UNet
from dataset import Dataset


def Loss(nn.Module)
    def __init__(self, coef: float=2.):
        super().__init__()
        self.coef = coef
        
    def forward(self, gt, pred):
        ind = gt < 0
        l1 = (pred[pred[ind] > 0] - 1).sum()
        
        ind = gt > 0
        l2 = (pred[ind] - gt[ind]).sum()
        
        return l2 + self.coef*l1
    
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
                
                with torch.set_grad_enabled(phase == 'train'):
                    pred = model(images)
                    error = loss(gt, pred)
                    
                    if phase == 'train':
                        error.backward()
                        
                    loss_per_epoch += error
                    
            print(epoch, phase, loss_per_epoch.mean())
        
        if phase == 'train':
            torch.save(model.state_dict(), f'models/{epoch}.pth')
            
def main():
    os.mkdirs('models', exist_ok=True)
    
    model = Doc_UNet()
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], 
        lr=1e-3, weight_decay = 1e-3, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=3, gamma=0.5)
    dataset = Dataset('dataset')
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [int(len(data)*0.8), int(len(data)*0.2)]
    )
    
    dataset = {'train': train_dataset, 'val': val_dataset}
    dataloader = {
        phase: torch.utils.data.DataLoader(
            dataset[phase], 
            batch_size = 4, 
            shuffle = True, 
            drop_last = True, 
            collate_fn = lambda batch: tuple(zip(*batch))) 
        for phase in dataset}
    
    train(model, optimizer, scheduler, dataloader, 10, device)
    
if __name__ == '__main__':
    main()