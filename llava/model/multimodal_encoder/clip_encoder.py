import torch
import torch.nn as nn

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from huggingface_hub import hf_hub_download
import json

def get_open_clip_image_processor(model_name):
    config_path = hf_hub_download(model_name, filename="open_clip_config.json")

    with open(config_path, 'r') as f:
        config = json.load(f)
    image_size = config['model_cfg']['vision_cfg']['image_size']
    image_mean = config['preprocess_cfg']['mean']
    image_std = config['preprocess_cfg']['std']
    size = {"shortest_edge": image_size}
    crop_size = {
      "height": image_size,
      "width": image_size
    }

    return CLIPImageProcessor(
            image_size=image_size, 
            image_mean=image_mean, 
            image_std=image_std,
            crop_size=crop_size,
            size=size
            )


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self):
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name)
        if self.vision_tower_name.startswith("apple") or self.vision_tower_name.startswith("laion"):
            self.image_processor = get_open_clip_image_processor(self.vision_tower_name)
        else:
            self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)

        self.vision_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
