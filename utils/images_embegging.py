import glob
import os
from torchvision.models.vision_transformer import vit_b_16
from torchvision.models.vision_transformer import ViT_B_16_Weights
from torchvision.io import ImageReadMode
from torchvision.io import read_image
from torchvision.transforms import v2
import torch

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

img_path = 'C:\\Users\\GRVRLD00P\\Documents\\Progetto ORS\\Dati\\ResizedImages\\*'
images = glob.glob(img_path)

vit = vit_b_16(ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
transform_img = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])

embeddings = []
ids = []
for path in images:
    image = transform_img(read_image(path, ImageReadMode.RGB)).unsqueeze(0)
    with torch.no_grad():
        img_emb = vit(image)
    embeddings.append(img_emb.squeeze())
    ids.append(os.path.basename(path))

img_dict = {}
for i in range(0,len(ids)):
   img_dict[ids[i]] = embeddings[i]

torch.save(img_dict, 'images.pt')