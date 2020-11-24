import torch, os, PIL, torchvision

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        len_masks = len(os.listdir(os.path.join(path, 'masks')))
        len_images = len(os.listdir(os.path.join(path, 'images')))
        
        if len(self.masks) != len(self.images)
            raise Exception(f'lengths of masks({len(self.masks)}) and images({len(self.images)}) arent the same')
            
        self.length = len_mask
        
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> tuple[torch.tensor]:
        
        image = os.path.join(self.path, 'images', f'{idx}.jpg')
        mask  = os.path.join(self.path, 'masks', f'{idx}.npy')
        
        for path in (image, mask):
            if not os.path.exists(path):
                raise Exception(f'Missing {path} in dataset')
                
        mask = torch.from_numpy(
            np.load(mask).astype(np.float32))
        image = torchvision.transforms.functional.to_tensor(
            PIL.Image.open(image).convert('RGB'))
        
        return image, mask
        