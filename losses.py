import torch
from kornia.losses import BinaryFocalLossWithLogits


def loss(model, name='BCE', balance=False, dataset=None, dataloader=None, device='cpu'):
    model.model_name += f'_{name}'
    pos_weight = None
    
    if balance:
        if dataset is None or dataloader is None:
            raise ValueError("balance is set to True, but the data is not given")
        
        # Compute weights for loss function
        num_labels = len(dataset[0]['targets'])
        pos_num = torch.zeros(num_labels).to(device)
        for _, data in enumerate(dataloader, 0):
            targets = data['targets'].to(device)
            pos_num += torch.sum(targets, axis=0)
        nobs = len(dataloader.dataset)
        pos_weight = (nobs - pos_num) / pos_num

        model.model_name += "-Balanced"
    
    if name == 'BCE':
        return torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif name == 'BinaryFocal':
        return BinaryFocalLossWithLogits(pos_weight=pos_weight, gamma=0.5, alpha=1, reduction='mean')
    else:
        raise ValueError("loss not known")
