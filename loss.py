import torch
from torchvision import models
from torch import nn, optim

class VGGLoss(nn.Module):
    def __init__(self):
        super(VGGLoss, self).__init__()
        #weight = models.VGG16_Weights.IMAGENET1K_V1
        # weight = models.VGG16_Weights.DEFAULT
        vgg = models.vgg16(pretrained=True)
        self.contentLayers = nn.Sequential(*list(vgg.features)[:31]).cuda().eval()
        for param in self.contentLayers.parameters():
            param.requires_grad = False

    def forward(self, pred, sharp):
        perceptual_loss = torch.nn.functional.mse_loss(self.contentLayers(pred), self.contentLayers(sharp))
        return perceptual_loss
    
class Keypoint_Loss(nn.Module):
    def __init__(self):
        super(Keypoint_Loss, self).__init__()
        self.MSE_loss = nn.MSELoss(reduction='none')
        
    def forward(self, pred, sharp, kpt):
        keypoint_loss_ = torch.mul(self.MSE_loss(pred, sharp), kpt)
        #if torch.count_nonzero(keypoint_loss_) == 0:
        #    keypoint_loss = 0
        #else:
        #    keypoint_loss = torch.sum(keypoint_loss_) / torch.count_nonzero(keypoint_loss_)
        keypoint_loss = torch.sum(keypoint_loss_) / torch.numel(keypoint_loss_)
        return keypoint_loss

    
