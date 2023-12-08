import os
from .clip_encoder import CLIPVisionTower
from .clap_encoder import CLAPAudioTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    is_absolute_path_exists = os.path.exists(vision_tower)
    if is_absolute_path_exists or vision_tower.startswith("openai") or vision_tower.startswith("laion") or vision_tower.startswith("apple"):
        return CLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

    raise ValueError(f'Unknown vision tower: {vision_tower}')

def build_audio_tower(audio_tower_cfg, **kwargs):
    audio_tower = getattr(audio_tower_cfg, 'mm_audio_tower', getattr(audio_tower_cfg, 'audio_tower', None))
    is_absolute_path_exists = os.path.exists(audio_tower)
    if is_absolute_path_exists or audio_tower.startswith("laion"):
        return CLAPAudioTower(audio_tower, args=audio_tower_cfg, **kwargs)

    raise ValueError(f'Unknown audio tower: {audio_tower}')
