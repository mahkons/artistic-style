import argparse
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import PIL
from tqdm import tqdm

DEFAULT_IMAGE_SIZE = (224, 224)

def create_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="vgg19", required=False, choices=["vgg11", "vgg13", "vgg16", "vgg19"])
    parser.add_argument("--content-image", type=str, default="./data/train.png", required=False)
    parser.add_argument("--style-image", type=str, default="./data/starry-night.jpg", required=False)
    parser.add_argument("--device", type=str, default="cpu", required=False, choices=["cpu", "cuda"])
    return parser 



def load_model(model_str):
    if model_str == "vgg11":
        return torchvision.models.vgg19(pretrained=True)
    elif model_str == "vgg13":
        return torchvision.models.vgg19(pretrained=True)
    elif model_str == "vgg16":
        return torchvision.models.vgg19(pretrained=True)
    elif model_str == "vgg19":
        return torchvision.models.vgg19(pretrained=True)
    else:
        assert(False)


_NORMALIZE_MEAN = torch.tensor([0.485, 0.456, 0.406])
_NORMALIZE_STD = torch.tensor([0.229, 0.224, 0.225])

def load_image(image_path, to_size, device):
    image = PIL.Image.open(image_path).convert("RGB")
    if to_size:
        image = torchvision.transforms.Resize(to_size)(image)

    image = torchvision.transforms.ToTensor()(image).to(dtype=torch.float, device=device)
    image.requires_grad_(False)

    image = (image - _NORMALIZE_MEAN[:, None, None].to(image.device)) / _NORMALIZE_STD[:, None, None].to(image.device) # preprocessing for torch pretrained models
    return image


def imshow(image):
    image = (image * _NORMALIZE_STD[:, None, None].to(image.device)) + _NORMALIZE_MEAN[:, None, None].to(image.device)
    image = torchvision.transforms.ToPILImage()(image.cpu())
    plt.imshow(image)
    return plt.show()






# features
# works only with vgg models =(
def extract_conv_features(model, image):
    i, j = 1, 1  # number of conv_layer
    image = image[None]

    features = list()
    for layer in model.features.children():
        if isinstance(layer, nn.Conv2d):
            image = layer(image)
        elif isinstance(layer, nn.ReLU):
            image = layer(image)
            features.append((i, j, image.clone().squeeze(0)))
        elif isinstance(layer, nn.MaxPool2d):
            image = layer(image)

            # net conv block
            i += 1
            j = 1
        else:
            assert(False)

    return features


def choose_features(features):
    features = filter(lambda f: f[1] == 1, features) # take first convolution from every layer
    features = map(lambda f: f[2], features)
    return list(features)


def get_gram_matrix(image):
    image = image.reshape(image.shape[0], -1).mean(dim=1, keepdims=True)
    return image @ image.T

def calc_loss(model, image, content_image, style_image):
    image_features = choose_features(extract_conv_features(model, image))
    content_image_features = choose_features(extract_conv_features(model, content_image))
    style_image_features = choose_features(extract_conv_features(model, style_image))

    content_loss, style_loss = 0, 0

    for image_f, content_image_f, style_image_f in zip(image_features, content_image_features, style_image_features):
        content_loss += F.mse_loss(image_f, content_image_f)

        gram_image = get_gram_matrix(image_f)
        gram_style_image = get_gram_matrix(style_image_f)
        style_loss += F.mse_loss(gram_image, gram_style_image)
        
    return content_loss * CONTENT_COEFF + style_loss * STYLE_COEFF


ITER = 100
LR = 1
CONTENT_COEFF = 1
STYLE_COEFF = 0

def transfer(model, content_image, style_image, device):
    image = torch.rand(content_image.shape, device=device)
    optimizer = torch.optim.Adam((image,), lr=LR)

    for i in tqdm(range(ITER)):
        loss = calc_loss(model, image, content_image, style_image)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        image = image.clamp(0, 1)
        

    return image

if __name__ == "__main__":
    args = create_parser().parse_args()
    device = torch.device(args.device)
    model = load_model(args.model).to(device)

    content_image = load_image(args.content_image, DEFAULT_IMAGE_SIZE, device)
    style_image = load_image(args.style_image, DEFAULT_IMAGE_SIZE, device)

    new_image = transfer(model, content_image, style_image, device)

    imshow(torchvision.utils.make_grid([style_image, content_image, new_image]))
