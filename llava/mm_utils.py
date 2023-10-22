from PIL import Image
from io import BytesIO
import base64

import torch
from transformers import StoppingCriteria
from llava.constants import IMAGE_TOKEN_INDEX


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result

"""
def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            # Break down image into 4 quadrants
            width, height = image.size
            patches = [
                expand2square(image.crop((0, 0, width//2, height//2)), tuple(int(x*255) for x in image_processor.image_mean)),  # top left
                expand2square(image.crop((width//2, 0, width, height//2)), tuple(int(x*255) for x in image_processor.image_mean)),  # top right
                expand2square(image.crop((0, height//2, width//2, height)), tuple(int(x*255) for x in image_processor.image_mean)),  # bottom left
                expand2square(image.crop((width//2, height//2, width, height)), tuple(int(x*255) for x in image_processor.image_mean))  # bottom right
            ]

            # Process each patch and original image
            patch_tensors = [image_processor.preprocess(patch, return_tensors='pt')['pixel_values'][0] for patch in patches]
            original_image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            original_image_tensor = image_processor.preprocess(original_image, return_tensors='pt')['pixel_values'][0]

            # Combine original image tensor with patch tensors
            combined_tensor = torch.cat([original_image_tensor] + patch_tensors, dim=0)
            new_images.append(combined_tensor)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            # Break down image into 4 quadrants
            width, height = image.size
            patches = [
                expand2square(image.crop((0, 0, width//2, height//2)), tuple(int(x*255) for x in image_processor.image_mean)),  # top left
                expand2square(image.crop((width//2, 0, width, height//2)), tuple(int(x*255) for x in image_processor.image_mean)),  # top right
                expand2square(image.crop((0, height//2, width//2, height)), tuple(int(x*255) for x in image_processor.image_mean)),  # bottom left
                expand2square(image.crop((width//2, height//2, width, height)), tuple(int(x*255) for x in image_processor.image_mean))  # bottom right
            ]

            # Process each patch and original image
            patch_tensors = [image_processor.preprocess(patch, return_tensors='pt')['pixel_values'][0] for patch in patches]
            original_image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            original_image_tensor = image_processor.preprocess(original_image, return_tensors='pt')['pixel_values'][0]

            # Combine original image tensor with patch tensors along the batch dimension
            combined_tensor = torch.cat([original_image_tensor.unsqueeze(0)] + [patch.unsqueeze(0) for patch in patch_tensors], dim=0)
            new_images.append(combined_tensor)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            # Break down image into 4 quadrants
            width, height = image.size
            patches = [
                expand2square(image.crop((0, 0, width//2, height//2)), tuple(int(x*255) for x in image_processor.image_mean)),  # top left
                expand2square(image.crop((width//2, 0, width, height//2)), tuple(int(x*255) for x in image_processor.image_mean)),  # top right
                expand2square(image.crop((0, height//2, width//2, height)), tuple(int(x*255) for x in image_processor.image_mean)),  # bottom left
                expand2square(image.crop((width//2, height//2, width, height)), tuple(int(x*255) for x in image_processor.image_mean))  # bottom right
            ]

            # Process each patch and original image
            patch_tensors = [image_processor.preprocess(patch, return_tensors='pt')['pixel_values'][0] for patch in patches]
            original_image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            original_image_tensor = image_processor.preprocess(original_image, return_tensors='pt')['pixel_values'][0]

            # Combine original image tensor with patch tensors along the batch dimension
            combined_tensor = torch.cat([original_image_tensor.unsqueeze(0)] + [patch.unsqueeze(0) for patch in patch_tensors], dim=0)
            new_images.append(combined_tensor)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            # Break down image into 4 quadrants
            width, height = image.size
            patches = [
                expand2square(image.crop((0, 0, width//2, height//2)), tuple(int(x*255) for x in image_processor.image_mean)),  # top left
                expand2square(image.crop((width//2, 0, width, height//2)), tuple(int(x*255) for x in image_processor.image_mean)),  # top right
                expand2square(image.crop((0, height//2, width//2, height)), tuple(int(x*255) for x in image_processor.image_mean)),  # bottom left
                expand2square(image.crop((width//2, height//2, width, height)), tuple(int(x*255) for x in image_processor.image_mean))  # bottom right
            ]

            # Process each patch and original image
            patch_tensors = [image_processor.preprocess(patch, return_tensors='pt')['pixel_values'][0] for patch in patches]
            original_image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            original_image_tensor = image_processor.preprocess(original_image, return_tensors='pt')['pixel_values'][0]

            # Resize patch tensors to match the original image tensor size
            patch_tensors_resized = [torch.nn.functional.interpolate(patch.unsqueeze(0), size=original_image_tensor.shape[1:], mode='bilinear').squeeze(0) for patch in patch_tensors]

            # Combine original image tensor with patch tensors along the batch dimension
            combined_tensor = torch.cat([original_image_tensor.unsqueeze(0)] + [patch.unsqueeze(0) for patch in patch_tensors_resized], dim=0)
            new_images.append(combined_tensor)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            width, height = image.size
            
            # Break down image into 4 quadrants
            patches = [
                image.crop((0, 0, width//2, height//2)),  # top left
                image.crop((width//2, 0, width, height//2)),  # top right
                image.crop((0, height//2, width//2, height)),  # bottom left
                image.crop((width//2, height//2, width, height))  # bottom right
            ]
            
            # Resize patches to match original image size
            patches = [patch.resize((width, height)) for patch in patches]

            # Process each patch and original image
            patch_tensors = [image_processor.preprocess(patch, return_tensors='pt')['pixel_values'][0] for patch in patches]
            original_image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]

            # Combine original image tensor with patch tensors along the batch dimension
            combined_tensor = torch.cat([original_image_tensor.unsqueeze(0)] + [patch.unsqueeze(0) for patch in patch_tensors], dim=0)
            new_images.append(combined_tensor)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            # Break down image into 4 quadrants
            width, height = image.size
            patches = [
                expand2square(image.crop((0, 0, width//2, height//2)), tuple(int(x*255) for x in image_processor.image_mean)),  # top left
                expand2square(image.crop((width//2, 0, width, height//2)), tuple(int(x*255) for x in image_processor.image_mean)),  # top right
                expand2square(image.crop((0, height//2, width//2, height)), tuple(int(x*255) for x in image_processor.image_mean)),  # bottom left
                expand2square(image.crop((width//2, height//2, width, height)), tuple(int(x*255) for x in image_processor.image_mean))  # bottom right
            ]

            # Process each patch and original image
            patch_tensors = [image_processor.preprocess(patch, return_tensors='pt')['pixel_values'][0] for patch in patches]
            original_image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            original_image_tensor = image_processor.preprocess(original_image, return_tensors='pt')['pixel_values'][0]

            # Combine original image tensor with patch tensors into a list
            combined_tensor = [original_image_tensor] + patch_tensors
            new_images.append(combined_tensor)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    return new_images
"""

"""
def split_image_into_quadrants(image):
    width, height = image.size
    quadrant_size = (width // 2, height // 2)
    quadrants = [
        image.crop((0, 0, quadrant_size[0], quadrant_size[1])),
        image.crop((quadrant_size[0], 0, width, quadrant_size[1])),
        image.crop((0, quadrant_size[1], quadrant_size[0], height)),
        image.crop((quadrant_size[0], quadrant_size[1], width, height))
    ]
    return quadrants

def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            # Break the image into 4 quadrants
            width, height = image.size
            quadrants = [image.crop((0, 0, width//2, height//2)), image.crop((width//2, 0, width, height//2)),
                         image.crop((0, height//2, width//2, height)), image.crop((width//2, height//2, width, height))]
            # Process each quadrant and the original image
            processed_images = [image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0] for img in [image] + quadrants]
            new_images.extend(processed_images)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images

def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            # Break the image into 4 quadrants
            width, height = image.size
            quadrants = [image.crop((0, 0, width//2, height//2)), image.crop((width//2, 0, width, height//2)),
                         image.crop((0, height//2, width//2, height)), image.crop((width//2, height//2, width, height))]
            # Process each quadrant and the original image
            processed_images = []
            for img in [image] + quadrants:
                img = expand2square(img, tuple(int(x*255) for x in image_processor.image_mean))
                img = img.resize((width, height))  # Resize the image to the original size
                processed_images.append(image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0])
            new_images.extend(processed_images)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images
"""
def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            # Break the image into 4 quadrants
            width, height = image.size
            quadrants = [image.crop((0, 0, width//2, height//2)), image.crop((width//2, 0, width, height//2)),
                         image.crop((0, height//2, width//2, height)), image.crop((width//2, height//2, width, height))]
            # Process each quadrant and the original image
            processed_images = []
            for img in [image] + quadrants:
                img = expand2square(img, tuple(int(x*255) for x in image_processor.image_mean))
                img = img.resize((width, height))  # Resize the image to the original size
                processed_img = image_processor.preprocess(img, return_tensors='pt')['pixel_values'][0]
                processed_images.append(processed_img)
                print(f"Image shape: {processed_img.shape}")  # Print the shape of each image
            new_images.extend(processed_images)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images
def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]




class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0]:] == keyword_id:
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
