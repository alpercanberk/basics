import torch
from PIL import Image
from typing import List
from transformers import AutoProcessor, AutoModel

"""
pip install protobuf transformers sentencepiece
"""

CACHE_DIR = '/local/vondrick/alper/hf/hub'
siglip_model = AutoModel.from_pretrained("google/siglip-so400m-patch14-384", cache_dir=CACHE_DIR)
siglip_processor = AutoProcessor.from_pretrained("google/siglip-so400m-patch14-384", cache_dir=CACHE_DIR)
siglip_model.to('cuda:1')

def get_clip_text_image_similarity(images: List[Image.Image], 
                                   text: List[str], 
                                   processor, 
                                   model, 
                                   device='cuda:1'):
    """
    Takes in M images and N texts and returns a MxN matrix of dot products between the image and text embeddings.
    """
    inputs = processor(text=text, images=images, return_tensors="pt").to(device)
    outputs = model(**inputs)

    image_features = outputs['image_embeds']
    text_features = outputs['text_embeds']

    dot_products = image_features @ text_features.T
    return dot_products

if __name__ == '__main__':
    images = [Image.open('transforms_0.png').convert('RGB')] #these are picutres on my own directory rn
    text = ['a photo of a cat', 'a photo of a dog', 'a photo of a robot'] 
    dot_products = get_clip_text_image_similarity(images, text, siglip_processor, siglip_model)
    print(dot_products)     #[0.0018, -0.0002, 0.0151]