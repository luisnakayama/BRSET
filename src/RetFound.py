import torch
import torch.nn as nn
#from timm.models.layers import trunc_normal_
import timm.models.vision_transformer
from functools import partial
import gdown
import os

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling

    Args:
        global_pool (bool): If True, use global average pooling. Default is False.
        **kwargs: Additional keyword arguments for the base VisionTransformer class.
    """
    def __init__(self, global_pool=False, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.global_pool = global_pool
        if self.global_pool:
            norm_layer = kwargs['norm_layer']
            embed_dim = kwargs['embed_dim']
            self.fc_norm = norm_layer(embed_dim)

            del self.norm  # remove the original norm

    def forward_features(self, x):
        """Forward pass through the feature extraction layers of the Vision Transformer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        return outcome


def vit_large_patch16(**kwargs):
    """Instantiate a Vision Transformer model with specific configuration.

    Returns:
        VisionTransformer: Instance of the VisionTransformer class.
    """
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# --------------------------------------------------------
# Interpolate position embeddings for high-resolution
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    """Interpolate position embeddings for high-resolution.

    Args:
        model (VisionTransformer): Instance of the VisionTransformer class.
        checkpoint_model (dict): Checkpoint model containing position embeddings.
    """
    if 'pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['pos_embed']
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.patch_embed.num_patches
        num_extra_tokens = model.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed


# Create a new model with the desired modifications
class ModifiedRetFound(nn.Module):
    """Modified version of the RetFound model for custom tasks.

    Args:
        backbone (nn.Module): Backbone model to be modified.
        num_classes (int): Number of output classes for the new classifier head.
    """
    def __init__(self, backbone, num_classes):
        super(ModifiedRetFound, self).__init__()

        self.num_classes = num_classes

        # Extract the backbone without the top layer
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])

        # New classifier head
        if self.num_classes:
            self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        """Forward pass through the modified RetFound model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Backbone
        x = self.backbone(x)

        # Average pooling
        x = x[:, 1:, :].mean(dim=1)  # global pool without cls token

        # Classifier head
        if self.num_classes:
            x = self.classifier(x)

        return x

def get_retfound(weights=None, num_classes=3, backbone=False):
    """Get the modified RetFound model with specified weights and configurations.

    Args:
        weights (str): Path to the pretrained weights file. If not provided, weights will be downloaded.
        num_classes (int): Number of output classes for the new classifier head.
        backbone (bool): If True, return only the backbone without the classifier head.

    Returns:
        ModifiedRetFound: Instance of the modified RetFound model.
    """
    # call the model
    model = vit_large_patch16(
        num_classes=3,
        drop_path_rate=0.2,
        global_pool=True
    )

    
    if not weights:
        download_weights = input("Do you want to download the pretrained weights? (y/n): ")
        if download_weights in ['y', 'Y', 'yes', 'Yes', 'YES']:
            # download RETFound weights
            output_file = 'Weights/RETFound_cfp_weights.pth'
            file_url = 'https://drive.google.com/uc?id=1l62zbWUFTlp214SvK6eMwPQZAzcwoeBE'
            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            # Download the file from the Google Drive link
            gdown.download(file_url, output_file, quiet=False)
            print(f"File '{output_file}' downloaded successfully.")
            weights = output_file
        else:
            print("Please provide the path to the pretrained weights file.")
            return

    # load RETFound weights
    checkpoint = torch.load(weights, map_location='cpu')
    checkpoint_model = checkpoint['model']
    state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias']:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del checkpoint_model[k]

    # interpolate position embeddings for high-resolution# interpolate position embedding
    interpolate_pos_embed(model, checkpoint_model)
    # load pre-trained model
    msg = model.load_state_dict(checkpoint_model, strict=False)

    assert set(msg.missing_keys) == {'head.weight', 'head.bias', 'fc_norm.weight', 'fc_norm.bias'}

    # manually initialize fc layer
    # trunc_normal_(model.head.weight, std=2e-5)

    # Define the number of classes for the new head
    if backbone:    
        num_classes = None
    
    # Instantiate the modified model
    model = ModifiedRetFound(model, num_classes)

    return model

