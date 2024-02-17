from torchvision import transforms
from torch import nn
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.backbone.state_dict(), self.path)
        self.val_loss_min = val_loss



def get_augmentations(image_size):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=image_size[0] // 20 * 2 + 1, sigma=(0.1, 2.0))], p=0.5),
    ])


def byol_loss(p1, p2, z1, z2):
    p1 = F.normalize(p1, dim=-1)
    z1 = F.normalize(z1, dim=-1)
    p2 = F.normalize(p2, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    return 2 - 2 * (F.cosine_similarity(p1, z2).mean() + F.cosine_similarity(p2, z1).mean()) / 2


def train_byol(byol_model, dataloader, eval_dataloader, epochs, optimizer, device, patience=10, path='checkpoint_byol.pt'):
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=path)
    
    byol_model.to(device)
    for epoch in range(epochs):
        train_loss = 0
        num_train_batches = len(dataloader)

        for batch in tqdm(dataloader, total=num_train_batches):
            
            # Create two augmented views of the same batch
            image_one, image_two = batch[0].to(device), batch[1].to(device)

            # Forward pass
            online_pred_one, online_pred_two, target_proj_one, target_proj_two = byol_model(image_one, image_two)

            # Loss
            loss = byol_loss(online_pred_one, online_pred_two, target_proj_one, target_proj_two)
            train_loss += loss.item()

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update target network
            byol_model.update_target_network()
        train_loss /= len(dataloader)
            
        with torch.no_grad():
            eval_loss = 0
            for val_batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
                # Create two augmented views of the same batch
                eval_image_one, eval_image_two = val_batch[0].to(device), val_batch[1].to(device)
                online_pred_one, online_pred_two, target_proj_one, target_proj_two = byol_model(eval_image_one, eval_image_two)
                eval_loss += byol_loss(online_pred_one, online_pred_two, target_proj_one, target_proj_two).item()
            eval_loss /= len(eval_dataloader)
                
            early_stopping(eval_loss, byol_model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print(f'Epoch {epoch+1}, Training Loss: {train_loss}, Validation Loss: {eval_loss}')

