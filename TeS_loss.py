import torch

def TeS_loss(text_feature, model_output, target, criterion, tau, lambda_v, lambda_t):
    output = model_output['x']
    x_proj = model_output['x_proj']
    
    logits_text = x_proj @ text_feature.T

    with torch.no_grad():
        soft_label = lambda_v * torch.softmax(logits_text/tau, dim=1)
        soft_label[torch.arange(len(target), device=target.device), target] += 1. - lambda_v

    loss_vision = criterion(output, soft_label)
    loss_text = criterion(logits_text/tau, target)
    loss = loss_vision + lambda_t * loss_text

    return loss