import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)

class BYOL(nn.Module):
    def __init__(self, backbone, image_size=224, hidden_dim=4096, projection_dim=256):
        super(BYOL, self).__init__()
        self.backbone = backbone
        self.backbone_output_dim = self.calculate_backbone_out()

        # Projector
        self.projector = MLP(self.backbone_output_dim, hidden_dim, projection_dim)

        # Predictor
        self.predictor = MLP(projection_dim, hidden_dim // 4, projection_dim)

        # Target network
        self.target_backbone = copy.deepcopy(backbone)
        self.target_projector = MLP(self.backbone_output_dim, hidden_dim, projection_dim)

        # Freeze the target networks
        for p in self.target_backbone.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    def forward(self, image_one, image_two):
        # Online network forward pass
        online_proj_one = self.projector(self.backbone(image_one))
        online_pred_one = self.predictor(online_proj_one)

        online_proj_two = self.projector(self.backbone(image_two))
        online_pred_two = self.predictor(online_proj_two)

        # Target network forward pass
        with torch.no_grad():
            target_proj_one = self.target_projector(self.target_backbone(image_one))
            target_proj_two = self.target_projector(self.target_backbone(image_two))

        return online_pred_one, online_pred_two, target_proj_one, target_proj_two

    def update_target_network(self, tau=0.99):
        # Update target network parameters with a slow-moving average
        for online_params, target_params in zip(self.backbone.parameters(), self.target_backbone.parameters()):
            target_params.data = tau * target_params.data + (1 - tau) * online_params.data
        for online_params, target_params in zip(self.projector.parameters(), self.target_projector.parameters()):
            target_params.data = tau * target_params.data + (1 - tau) * online_params.data
            
            
    def calculate_backbone_out(self):
        sample_input = torch.randn(1, 3, 224, 224)
        
        self.backbone.eval()
        # Forward pass the sample input through the model
        with torch.no_grad():
            output = self.backbone(sample_input)
        return output.shape[1]
        
