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
from tqdm import tqdm
from torchvision.datasets import (CIFAR10, CIFAR100)
from datetime import datetime


class CLIP_Encoder():
    """
    OpenCLIP encoder
    """
    def __init__(self, config=None, dataset="cifar10"):
        self.model_name = config['model']
        self.pretrained = config['pretrained']
        self.dataset_name = dataset
        self.cache_path = config['cache_path']
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.dataset_info = {
            "dataset": "",
            "num_classes": 0,
            "num_images": 0,
            "accuracy": 0,
            "correct": 0,
            "total": 0,
            "per_class_accuracy": 0,
            "class_names": "",
            "template": ""
        }
        self.dataset_results = {
            "model": "",
            "pretrained": "",
            "num_params": 0,
            "datasets": {}
        }

    def init(self):
        print("CLIP Encoder!")
        print("\n" + "="*60)
        print(f"Model: {self.model_name} ({self.pretrained})")
        print("="*60)

        # Load model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

    def encode_frame(self, frame_array):
        '''
        Step 1: CHW -> HWC
        CHW is converted to HWC so PIL can read the array.  frame_array
        is (3, H, W) and PIL expects (H, W, 3).  Numpy's transpose function
        re-orders axes to this form.  1->0(H) 2->1(W) 0->2(C).

        Step 2: numpy -> PIL

        Step 3: PIL -> Tensor
        PIL is converted to a pre-processed tensor [3, 224, 224].
        self.preprocess resizes, center-crops, coverts to float, and normalizes.
        .unsqueeze(0) adds the batch dimension -> [1, 3, 224, 224]

        Step 4: Run the frozen encoder without gradient tracking.
        We L2 normalize because dividing the vectors magnitude makes its length
        exactly 1.  The dot product between two vectors equals their cosine
        similarity.  This matches direction instead of magnitude.  CLIP is
        trained this way so comparisons only make sense in this normalized space.

        Note: Pytorch normally builds a computation graph while running to
        calculate gradients for backprop.  Since CLIP is frozen (meaning weights
        do not get updated) we do not need this graph.  To save memory,
        and time, we use no_grad() to turn this off.

        Step 5: Remove the fake batch dimension -> [512].
        '''
        # Step 1
        hwc = frame_array.transpose(1, 2, 0)

        # Step 2
        pil_image = Image.fromarray(hwc.astype(np.uint8))

        # Step 3
        pixel_values = self.preprocess(pil_image).unsqueeze(0)

        # Step 4
        with torch.no_grad():
            frame_emb = self.model.encode_image(pixel_values) # [1, 512]
            frame_emb = frame_emb / frame_emb.norm(dim=-1, keepdim=True) # L2 Normalize

        # Step 5
        return frame_emb.squeeze(0)

    def encode_subgoal(self, text):
        # CLIP's tokenizer expects a list of strings.  It converts words
        # into integer token IDs that the next transformer understands.
        tokens = self.tokenizer([text])

        with torch.no_grad():
            goal_emb = self.model.encode_text(tokens) # [1, 512]
            goal_emb = goal_emb / goal_emb.norm(dim=-1, keepdim=True)

        return goal_emb.squeeze(0) # [512]

    def load_cifar10(self):
        """
        Cifar-10: 10 classes, 10,000 test images.
        """
        dataset = CIFAR10(
            root=self.cache_path,
            train=False,
            download=True,
            transform=self.preprocess
        )
        class_names = dataset.classes # ['airplane', 'automobile', 'bird', ...]
        template = "a photo of a {}"  # a photo of a {}, a type of flower
        return dataset, class_names, template

    def get_dataset(self, name):
        """
        Load dataset by name.
        """
        loaders = {
            "cifar10": self.load_cifar10
        }
        if name not in loaders:
            raise ValueError(f"Unknown dataset: {name}. " +
                             f"Choose from {list(loaders.keys())}")
        return loaders[name]()

    def encode_text_prompts(self, class_names, template):
        """
        Encode all class names into text embeddings.
        """
        prompts = [template.format(name) for name in class_names]
        text_tokens = self.tokenizer(prompts)

        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features

    def evaluate_dataset(self, batch_size=64):
        """
        Evaluate CLIP zero-shot accuracy on a dataset.
        """
        # Load dataset
        dataset_images, class_names, template = self.get_dataset(self.dataset_name)
        print(f"\n\tDataset: {self.dataset_name}")
        print(f"\tClasses: {len(class_names)}")
        print(f"\tNum Images: {len(dataset_images)}")
        print(f"\tImages: {dataset_images}")
        print(f"\tTemplate: \"{template}\"")

        # Encode text prompts for all classes
        text_features = self.encode_text_prompts(class_names, template)

        # Track predictions
        correct = 0
        total = 0
        per_class_correct = {i: 0 for i in range (len(class_names))}
        per_class_total = {i: 0 for i in range (len(class_names))}

        # Process images in batches
        dataloader = torch.utils.data.DataLoader(
            dataset_images, batch_size=batch_size, shuffle=False, num_workers=0
        )

        print(f"\tEvaluating...")
        for images, labels in tqdm(dataloader, desc=f"{self.dataset_name}"):
            with torch.no_grad():
                # Encode images
                # array Shape: torch.Size([64, 3, 224, 224]) B, C, H, W
                # print("Array Shape:", images.shape)
                # Data Type: torch.float32
                # print("Data Type:", images.dtype)
                image_features = self.model.encode_image(images)
                image_features = (
                    image_features / image_features.norm(dim=-1, keepdim=True)
                )

                # Compute similarity (images x classes)
                similarity = (100.0 * image_features @ text_features.T)

                # Get predictions (highest similarity)
                predictions = similarity.argmax(dim=-1)

            # Update counts
            for pred, label in zip(predictions, labels):
                label_idx = label.item()
                pred_idex = pred.item()

                per_class_total[label_idx] += 1
                if pred_idex == label_idx:
                    correct += 1
                    per_class_correct[label_idx] += 1
                total += 1

        # Compute accuracy
        accuracy = 100.0 * correct / total

        # Compute per-class accuracy
        per_class_accuracy = {}
        for i, name in enumerate(class_names):
            if per_class_total[i] > 0:
                per_class_accuracy[name] = (
                    100.0 * per_class_correct[i] / per_class_total[i]
                )
            else:
                per_class_accuracy[name] = 0.0

        print(f"\tAccuracy: {accuracy:.2f}%")

        self.dataset_info["name"] = self.dataset_name
        self.dataset_info["num_classes"] = len(class_names)
        self.dataset_info["num_images"] = len(dataset_images)
        self.dataset_info["accuracy"] = accuracy
        self.dataset_info["correct"] = correct
        self.dataset_info["total"] = total
        self.dataset_info["per_class_accuracy"] = per_class_accuracy
        self.dataset_info["class_names"] = class_names
        self.dataset_info["template"] = template

    def evaluate_model(self):
        """
        Evaluate a single CLIP model on multiple datasets.
        """
        # Print model info
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"Params: {num_params:,}")

        # Evaluate on each dataset
        self.dataset_results["model"] = self.model_name
        self.dataset_results["pretrained"] = self.pretrained
        self.dataset_results["num_params"] = num_params

        try:
            self.evaluate_dataset()
            self.dataset_results["datasets"][self.dataset_name] = self.dataset_info
        except Exception as e:
            print(f"\tError on {self.dataset_name}: {e}")
            self.dataset_results["datasets"][self.dataset_name] = {"error": str(e)}

    def print_summary_table(self, all_results):
        """
        Print a summary table of results.
        """
        print("\n" + "="*80)
        print("SUMMARY: Zero-Shot Accuracy (%)")
        print("="*80)

        # Header
        all_datasets = list(all_results[0]["datasets"].keys())
        header = f"{'Model:':<25} | " + " | ".join(f"{d:>10}" for d in all_datasets)
        print(header)
        print("-" * len(header))

        # Rows
        for result in all_results:
            model_name = result["model"]
            accuracy = []
            for ds in all_datasets:
                if "accuracy" in result["datasets"][ds]:
                    accuracy.append(f"{result['datasets'][ds]['accuracy']:>10.2f}")
                else:
                    accuracy.append(f"{'ERROR':>10}")
                row = f"{model_name:<25} | " + " | ".join(accuracy)
                print(row)

        print("="*80)

    def run_open_clip(self):
        """
        Run the full evaluation.
        """
        print(f"CLIP Zero-Shot Evaluation")
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Datasets: {self.dataset_name}")
        print(f"Models: {self.model_name}")

        # Evaluate each model
        self.evaluate_model()

        all_results = []
        all_results.append(self.dataset_results)

        # Print summary
        self.print_summary_table(all_results)

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
