"""
Zachary Meisner
Machine Learning: Deep Learning
EN.605.740.81.SU26
Dr. Alhassan S. Yasin

OpenCLIP encoder
clip_encoder.py
"""

import numpy as np
import torch
import open_clip
from open_clip import tokenizer
import matplotlib.pyplot as plt
from skimage import data, data_dir
import os
from PIL import Image
from torchvision.datasets import (CIFAR10, CIFAR100)

class CLIP_Encoder():
    """
    OpenCLIP encoder
    """
    def __init__(self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k"):
        self.model_name = model_name
        self.pretrained = pretrained
        self.model = None
        self.preprocess = None
        self.tokenizer = None


    def init(self):
        print("CLIP Encoder!")


    def test_clip(self):
        # Load model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self.model.eval()

        descriptions = {
            "page" : "a page of text about segmentation",
            "chelsea": "a facial photo of a tabby cat",
            "astronaut": "a portrait of an astronaut with the American flag",
            "rocket": "a rocket standing on a launchpad",
            "motorcycle_right": "a red motorcycle standing in a garage",
            "camera": "a person looking at a camera on a tripod",
            "horse": "a black-and-white silhouette of a horse",
            "coffee": "a cup of coffee on a saucer"
        }

        original_images = []
        images = []
        texts = []

        for filename in [filename for filename in os.listdir(data_dir)
                         if filename.endswith(".png")
                         or filename.endswith(".jpg")]:
            name = os.path.splitext(filename)[0]
            if name not in descriptions:
                continue

            image = Image.open(os.path.join(data_dir, filename)).convert("RGB")

            original_images.append(image)
            images.append(self.preprocess(image))
            texts.append(descriptions[name])

        image_input = torch.tensor(np.stack(images))
        text_tokens = tokenizer.tokenize(["This is " + desc for desc in texts])

        with torch.no_grad():
            image_features = self.model.encode_image(image_input).float()
            text_features = self.model.encode_text(text_tokens).float()

        #
        # Calculating cosine similarity
        #
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T
        count = len(descriptions)

        plt.figure(figsize=(20,14))
        plt.imshow(similarity, vmin=0.1, vmax=0.3)
        plt.yticks(range(count), texts, fontsize=18)
        plt.xticks([])

        for i, image in enumerate(original_images):
            plt.imshow(image, extent=(i-0.5, i+0.5, -1.6, -0.6), origin="lower")
        for x in range(similarity.shape[1]):
            for y in range(similarity.shape[0]):
                plt.text(x, y, f"{similarity[y,x]:.2f}", ha="center",
                         va="center", size=12)

        for side in ["left", "top", "right", "bottom"]:
            plt.gca().spines[side].set_visible(False)

        plt.xlim([-0.5, count -0.5])
        plt.ylim([count+0.5, -2])
        plt.title("Cosine similarity between text and image features", size=20)

        plt.show()


    def test_zero_shot(self):
        # Load model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self.model.eval()

        descriptions = {
            "page" : "a page of text about segmentation",
            "chelsea": "a facial photo of a tabby cat",
            "astronaut": "a portrait of an astronaut with the American flag",
            "rocket": "a rocket standing on a launchpad",
            "motorcycle_right": "a red motorcycle standing in a garage",
            "camera": "a person looking at a camera on a tripod",
            "horse": "a black-and-white silhouette of a horse",
            "coffee": "a cup of coffee on a saucer"
        }

        original_images = []
        images = []
        texts = []

        for filename in [filename for filename in os.listdir(data_dir)
                         if filename.endswith(".png")
                         or filename.endswith(".jpg")]:
            name = os.path.splitext(filename)[0]
            if name not in descriptions:
                continue

            image = Image.open(os.path.join(data_dir, filename)).convert("RGB")

            original_images.append(image)
            images.append(self.preprocess(image))
            texts.append(descriptions[name])

        image_input = torch.tensor(np.stack(images))
        text_tokens = tokenizer.tokenize(["This is " + desc for desc in texts])

        with torch.no_grad():
            image_features = self.model.encode_image(image_input).float()
            text_features = self.model.encode_text(text_tokens).float()

        #
        # Calculating cosine similarity
        #
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        cifar100 = CIFAR100(os.path.expanduser("~/.cache"),
                            transform=self.preprocess, download=True)
        text_descriptions = [f"A photo of a {label}"
                             for label in cifar100.classes]
        text_tokens = tokenizer.tokenize(text_descriptions)

        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens).float()
            text_features /= text_features.norm(dim=-1, keepdim=True)

        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)

        plt.figure(figsize=(16,16))

        for i, image in enumerate(original_images):
            plt.subplot(4,4,2*i+1)
            plt.imshow(image)
            plt.axis("off")

            plt.subplot(4,4,2*i+2)
            y = np.arange(top_probs.shape[-1])
            plt.grid()
            plt.barh(y,top_probs[i])
            plt.gca().invert_yaxis()
            plt.gca().set_axisbelow(True)
            plt.yticks(y, [cifar100.classes[index]
                           for index in top_labels[i].numpy()])
            plt.xlabel("probability")

        plt.subplots_adjust(wspace=0.5)
        plt.show()
