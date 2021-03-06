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
    parser.add_argument("--content-image", type=str, default="./data/ekb.jpg", required=False)
    parser.add_argument("--style-image", type=str, default="./data/scream.jpg", required=False)
    parser.add_argument("--output-image", type=str, default="./generated/scream_ekb_1_vgg11.jpg", required=False)
    parser.add_argument("--device", type=str, default="cpu", required=False, choices=["cpu", "cuda"])
    return parser 


def load_model(model_str):
    if model_str == "vgg11":
        return torchvision.models.vgg11(pretrained=True)
    elif model_str == "vgg13":
        return torchvision.models.vgg13(pretrained=True)
    elif model_str == "vgg16":
        return torchvision.models.vgg16(pretrained=True)
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
    image.clamp_(0, 1)
    image = torchvision.transforms.ToPILImage()(image.cpu())
    plt.imshow(image)
    plt.show()

def imsave(image, path):
    image = (image * _NORMALIZE_STD[:, None, None].to(image.device)) + _NORMALIZE_MEAN[:, None, None].to(image.device)
    image = torchvision.transforms.ToPILImage()(image.cpu())
    image.save(path)


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
            j += 1
        elif isinstance(layer, nn.MaxPool2d):
            image = layer(image)

            # net conv block
            i += 1
            j = 1
        else:
            assert(False)

    return features


def detach_all(tensors):
    return [tensor.detach() for tensor in tensors]


def choose_content_features(features):
    features = filter(lambda f: f[0] == 4 and f[1] == 1, features) # take first convolution from 4th block
    features = map(lambda f: f[2], features)
    return list(features)


def choose_style_features(features):
    features = filter(lambda f: f[1] == 1, features) # take first convolution from every block
    features = map(lambda f: f[2], features)
    return list(features)


def get_gram_matrix(image):
    image = image.reshape(image.shape[0], -1)
    return (image @ image.T) / image.shape[1]

def calc_loss(model, image, content_image_features, style_image_features):
    image_features = extract_conv_features(model, image)
    new_content_features = choose_content_features(image_features)
    new_style_features = choose_style_features(image_features)

    content_loss, style_loss = 0, 0

    for image_f, content_image_f in zip(new_content_features, content_image_features):
        content_loss += F.mse_loss(image_f, content_image_f)

    for image_f, style_image_f in zip(new_style_features, style_image_features):
        gram_image = get_gram_matrix(image_f)
        gram_style_image = get_gram_matrix(style_image_f)
        style_loss += F.mse_loss(gram_image, gram_style_image)
        
    return content_loss, style_loss


ITER = 1000
LR = 1e-1
CONTENT_COEFF = 1
STYLE_COEFF = 1

def transfer(model, content_image, style_image, device):
    image = torch.rand(content_image.shape, requires_grad=True, device=device)
    optimizer = torch.optim.Adam((image,), lr=LR)

    content_image_features = detach_all(choose_content_features(extract_conv_features(model, content_image)))
    style_image_features = detach_all(choose_style_features(extract_conv_features(model, style_image)))

    pbar = tqdm(range(ITER))
    for i in pbar:
        content_loss, style_loss = calc_loss(model, image, content_image_features, style_image_features)
        pbar.write("content_loss: {}, style_loss: {}".format(content_loss * CONTENT_COEFF, style_loss * STYLE_COEFF))
        loss = content_loss * CONTENT_COEFF + style_loss * STYLE_COEFF
        with torch.no_grad():
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return image

if __name__ == "__main__":
    args = create_parser().parse_args()
    device = torch.device(args.device)
    model = load_model(args.model).to(device)

    content_image = load_image(args.content_image, DEFAULT_IMAGE_SIZE, device)
    style_image = load_image(args.style_image, DEFAULT_IMAGE_SIZE, device)

    new_image = transfer(model, content_image, style_image, device)

    im_grid = torchvision.utils.make_grid([style_image, content_image, new_image])
    imshow(im_grid)
    imsave(im_grid, args.output_image)
