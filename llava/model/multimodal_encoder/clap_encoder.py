import torch
import torch.nn as nn

from transformers import ClapModel, ClapProcessor, ClapAudioConfig, ClapAudioModel, AutoProcessor, AutoConfig


class CLAPAudioTower(nn.Module):
    def __init__(self, audio_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.audio_tower_name = audio_tower
        self.select_layer = args.mm_audio_select_layer
        # self.select_feature = getattr(args, 'mm_audio_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = ClapConfig.from_pretrained(self.audio_tower_name)

    def load_model(self):
        self.audio_tower = ClapAudioModel.from_pretrained(self.audio_tower_name)
        setattr(self.audio_tower,"config", self.audio_tower.audio_encoder.config)
        self.audio_processor = ClapProcessor.from_pretrained(self.audio_tower_name)
        self.audio_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, audio_forward_outs):
        audio_features = audio_forward_outs.hidden_states[self.select_layer]
        _, embed_dim, _, _ = audio_features.shape
        return audio_features.reshape(-1, embed_dim)

    @torch.no_grad()
    def forward(self, audios):
        audios, is_longer = audios['input_features'], audios['is_longer']
        audio_forward_outs = self.audio_tower(audios.to(device=self.device, dtype=self.dtype), is_longer=is_longer,output_hidden_states=True)
        audio_features = self.feature_select(audio_forward_outs).to(audios.dtype)

        return audio_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.audio_tower.dtype

    @property
    def device(self):
        return self.audio_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.audio_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        img_size = (self.config.spec_size, self.config.spec_size)
        patch_size = (
            (self.config.patch_size, self.config.patch_size) if isinstance(self.config.patch_size, int) else config.patch_size
        )
        patch_stride = (
            (self.config.patch_stride, self.config.patch_stride) if isinstance(self.config.patch_stride, int) else self.config.patch_stride
        )

        self.img_size = img_size
        self.patch_stride = patch_stride

        self.grid_size = (img_size[0] // patch_stride[0], img_size[1] // patch_stride[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        return self.grid_size[0] * self.grid_size[1]
