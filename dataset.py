import torch, os, torchvision, numpy as np, cv2


def resize_mask_and_image(path_mask, path_image, width, height):
    mask = np.load(path_mask).astype(np.float32)

    new_mask = np.ones((mask.shape[0], mask.shape[1], 3)) * (-1)
    new_mask[:, :, 0] = mask[:, :, 0]
    new_mask[:, :, 1] = mask[:, :, 1]

    resized_mask = cv2.resize(new_mask, (width, height), cv2.INTER_LANCZOS4)
    resized_mask = resized_mask[:, :, :2:]

    resized_image = cv2.resize(cv2.imread(path_image), (width, height), interpolation = cv2.INTER_LANCZOS4)

    return resized_mask, resized_image

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        len_masks = os.listdir(os.path.join(path, 'masks'))
        len_images = os.listdir(os.path.join(path, 'images'))
        
        if len(len_masks) != len(len_images):
            raise Exception(f'lengths of masks({len(len_masks)}) and images({len(len_images)}) arent the same')
            
        self.length = len_masks
        self.preprocess = torchvision.transforms.ToTensor()
        
    def __len__(self) -> int:
        return len(self.length)
    
    def __getitem__(self, idx: int):
        
        global_idx = int(self.length[idx].split('.')[0])
        image = os.path.join(self.path, 'images', f'{global_idx}.jpg')
        mask  = os.path.join(self.path, 'masks', f'{global_idx}.npy')
        
        for path in (image, mask):
            if not os.path.exists(path):
                raise Exception(f'Missing {path} in dataset')
                
        mask, image = resize_mask_and_image(mask, image, 272*3, 352*3)
                
        mask = torch.from_numpy(mask.transpose(2,0,1))
        image = torchvision.transforms.functional.to_tensor(image)
        
        return image, mask
    