"""
Zachary Meisner
525.733.8VL
Prof. Nasser Nasrabadi

Midterm - OpenCLIP Zero-Shot
https://colab.research.google.com
/github/mlfoundations/open_clip/blob/master/docs
/Interacting_with_open_clip.ipynb#scrollTo=IBRVTY9lbGm8

Evaluate pre-trained CLIP models on 5 datasets:
1. CIFAR-10 (baseline, 10 classes)
2. CIFAR-100 (fine-grained, 100 classes)
3. Flowers-102 (domain-specific)
4. Oxford-IIIT Pets (fine-grained animals)
5. EuroSAT (satellite imagery - curveball)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "" # Force CPU instead of GPU

import pandas as pd
import skimage # scikit-image
import torch
import open_clip
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
from datetime import datetime

# Dataset imports
from torchvision.datasets import (CIFAR10, CIFAR100, Flowers102,
                                  OxfordIIITPet, EuroSAT)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def runOpenClipTutorial():
    #
    # Loading the model
    #

    open_clip.list_pretrained()

    model, _, preprocess = open_clip.create_model_and_transforms(
        'convnext_base_w', pretrained='laion2b_s13b_b82k_augreg')
    # model in train mode by default, impacts some models with BatchNorm
    # or stochastic depth active
    model.eval()
    context_length = model.context_length
    vocab_size = model.vocab_size

    print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
    print("Context length:", context_length)
    print("Vocab size:", vocab_size)

    #
    # Text preprocessing
    #
    tokenizer = open_clip.get_tokenizer('convnext_base_w')
    tokenizer.tokenize("Hello World!")

    # images in skimage to use and their textual descriptions
    descriptions = {
        "page": "a page of text about segmentation",
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
    plt.figure(figsize=(16,5))

    for filename in [filename for filename in os.listdir(skimage.data_dir)
                     if filename.endswith(".png") or filename.endswith(".jpg")]:
        name = os.path.splitext(filename)[0]
        if name not in descriptions:
            continue

        image = Image.open(os.path.join(skimage.data_dir, filename)).convert("RGB")

        plt.subplot(2,4,len(images)+1)
        plt.imshow(image)
        plt.title(f"{filename}\n{descriptions[name]}")
        plt.xticks([])
        plt.yticks([])

        original_images.append(image)
        images.append(preprocess(image))
        texts.append(descriptions[name])

    plt.tight_layout()

    #
    # Building features
    #

    image_input = torch.tensor(np.stack(images))
    text_tokens = tokenizer.tokenize(["This is " + desc for desc in texts])

    with torch.no_grad():
        image_features = model.encode_image(image_input).float()
        text_features = model.encode_text(text_tokens).float()

    #
    # Calculating cosine similarity
    #
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = text_features.cpu().numpy() @ image_features.cpu().numpy().T

    count = len(descriptions)

    plt.figure(figsize=(20,14))
    plt.imshow(similarity, vmin=0.1, vmax=0.3)
    # plt.colorbar()
    plt.yticks(range(count), texts, fontsize=18)
    plt.xticks([])
    for i, image in enumerate(original_images):
        plt.imshow(image, extent=(i-0.5, i+0.5, -1.6, -0.6), origin="lower")
    for x in range(similarity.shape[1]):
        for y in range(similarity.shape[0]):
            plt.text(x, y, f"{similarity[y,x]:.2f}", ha="center", va="center", size=12)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, count -0.5])
    plt.ylim([count+0.5, -2])
    plt.title("Cosine similarity between text and image features", size=20)

    #
    # Zero-Shot Image Classification
    #

    cifar100 = CIFAR100(os.path.expanduser("~/.cache"), transform=preprocess,
                        download=True)

    text_descriptions = [f"A photo of a {label}" for label in cifar100.classes]
    text_tokens = tokenizer.tokenize(text_descriptions)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)

    text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    top_probs, top_labels = text_probs.cpu().topk(5, dim=-1)

    plt.figure(figsize=(16,16))

    for i, image in enumerate(original_images):
        plt.subplot(4, 4, 2 * i + 1)
        plt.imshow(image)
        plt.axis("off")

        plt.subplot(4, 4, 2 * i + 2)
        y = np.arange(top_probs.shape[-1])
        plt.grid()
        plt.barh(y, top_probs[i])
        plt.gca().invert_yaxis()
        plt.gca().set_axisbelow(True)
        plt.yticks(y, [cifar100.classes[index]
                       for index in top_labels[i].numpy()])
        plt.xlabel("probability")

    plt.subplots_adjust(wspace=0.5)
    plt.show()

# runOpenClipTutorial()



#
# Configuration
#
# Models to evaluate (architecture, pretrained weights)
MODELS = [
    ("ViT-B-32", "laion2b_s34b_b79k"), # Smaller ViT - baseline
    ("ViT-L-14", "laion2b_s32b_b82k"), # Larger ViT   - larger size impact
    ("convnext_base_w", "laion2b_s13b_b82k_augreg"), # ConvNet architecture
]

# Datasets to evaluate
DATASETS = ["cifar10", "cifar100", "flowers102", "pets", "eurosat"]

# Where to cache downloaded datasets
DATA_DIR = os.path.expanduser("~/.cache/clip_eval_data")

# Where to save results
RESULTS_DIR = "./results"

#
# Dataset Loaders - returns (dataset, class_names, prompt_template)
#

def load_cifar10(preprocess):
    """
    Cifar-10: 10 classes, 10,000 test images.
    """
    dataset = CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=preprocess
    )
    class_names = dataset.classes # ['airplane', 'automobile', 'bird', ...]
    template = "a photo of a {}"
    return dataset, class_names, template


def load_cifar100(preprocess):
    """
    Cifar-100: 100 fine-grained classes, 10,000 test images.
    """
    dataset = CIFAR100(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=preprocess
    )
    class_names = dataset.classes
    template = "a photo of a {}"
    return dataset, class_names, template


def load_flowers102(preprocess):
    """
    Flowers-102: 102 flower species, ~6,149 test images.
    """
    dataset = Flowers102(
        root=DATA_DIR,
        split="test",
        download=True,
        transform=preprocess
    )
    # Flowers102 does not include class names so we must define them.
    # Source: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/categories.html
    class_names = [
        "pink primrose", "hard-leaved pocket orchid", "canterbury bells",
        "sweet pea", "english marigold", "tiger lily", "moon orchid",
        "bird of paradise", "monkshood", "globe thistle", "snapdragon",
        "colt's foot", "king protea", "spear thistle", "yellow iris",
        "globe flower", "purple coneflower", "peruvian lily", "balloon flower",
        "giant white arum lily", "fire lily", "pincushion flower", "fritillary",
        "red ginger", "grape hyacinth", "corn poppy",
        "prince of wales feathers", "steamless gentian", "artichoke",
        "sweet william", "carnation", "garden phlox", "love in the mist",
        "mexican aster", "alpine sea holly", "ruby-lipped cattleya",
        "cape flower", "great masterwort", "siam tulip", "lenten rose",
        "barbeton daisy", "daffodil", "sword lily", "poinsettia",
        "bolero deep blue", "wallflower", "marigold", "buttercup",
        "oxeye daisy", "common dandelion", "petunia", "wild pansy", "primula",
        "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium",
        "orange dahlia", "pink-yellow dahlia", "cautleya spicata",
        "japanese anemone", "black-eyed susan", "silverbush",
        "californian poppy", "osteospermum", "spring crocus", "bearded iris",
        "windflower", "tree poppy", "gazania", "azalea", "water lily", "rose",
        "thorn apple", "morning glory", "passion flower", "lotus", "toad lily",
        "anthurium", "frangipani", "clematis", "hibiscus", "columbine",
        "desert-rose", "tree mallow", "magnolia", "cyclamen", "watercress",
        "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove",
        "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia",
        "blanket flower", "trumpet creeper", "blackberry lily"
    ]
    template = "a photo of a {}, a type of flower"
    return dataset, class_names, template


def load_pets(preprocess):
    """
    Oxford-IIIT Pets: 37 pet breeds (25 dogs, 12 cats), ~3,669 test images.
    """
    dataset = OxfordIIITPet(
        root=DATA_DIR,
        split="test",
        download=True,
        transform=preprocess
    )
    class_names = dataset.classes # ['Abyssinian', 'american_bulldog', ...]
    # Clean up class names (replace underscores with spaces).
    class_names = [name.replace("_", " ") for name in class_names]
    template = "a photo of a {}, a type of pet"
    return dataset, class_names, template


def load_eurosat(preprocess):
    """
    EuroSAT: 10 land use classes from satellite imagery, ~5,400 test images.
    """
    # EuroSAT doesn't have a built-in train/test split, so we use the
    # full dataset and take a subset for evaluation.
    dataset = EuroSAT(
        root=DATA_DIR,
        download=True,
        transform=preprocess
    )
    class_names = [
        "annual crop land", "forest", "herbaceous vegetation land",
        "highway", "industrial building", "pasture land", "permanent crop land",
        "residential building", "river", "sea or lake"
    ]
    template = "a satellite photo of a {}"
    return dataset, class_names, template


def get_dataset(name, preprocess):
    """
    Factory function to load a dataset by name.
    """
    loaders = {
        "cifar10": load_cifar10,
        "cifar100": load_cifar100,
        "flowers102": load_flowers102,
        "pets": load_pets,
        "eurosat": load_eurosat,
    }
    if name not in loaders:
        raise ValueError(f"Unknown dataset: {name}. " +
                         f"Choose from {list(loaders.keys())}")
    return loaders[name](preprocess)


#
# Evaluation Functions
#

def encode_text_prompts(model, tokenizer, class_names, template):
    """
    Encode all class names into text embeddings.

    Args:
        model: CLIP model.
        tokenizer: CLIP tokenizer.
        class_names: List of class names.
        template: Prompt template.

    Returns:
        Normalized text feature tensor (num_classes, embed_dim).
    """
    prompts = [template.format(name) for name in class_names]
    text_tokens = tokenizer(prompts)

    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    return text_features


def evaluate_dataset(model, preprocess, tokenizer, dataset_name, batch_size=64):
    """
    Evaluate CLIP zero-shot accuracy on a dataset.

    Args:
        model: CLIP model (in eval mode).
        preprocess: Image preprocessing transform.
        tokenizer: CLIP tokenizer.
        dataset_name: Name of dataset to evaluate.
        batch_size: Batch size for processing images.

    Returns:
        Dictionary with accuracy and per-class metrics.
    """
    # Load dataset
    dataset, class_names, template = get_dataset(dataset_name, preprocess)
    print(f"\n\tDataset: {dataset_name}")
    print(f"\tClasses: {len(class_names)}")
    print(f"\tImages: {len(dataset)}")
    print(f"\tTemplate: \"{template}\"")

    # Encode text prompts for all classes
    text_features = encode_text_prompts(model, tokenizer, class_names, template)

    # Track predictions
    correct = 0
    total = 0
    per_class_correct = {i: 0 for i in range(len(class_names))}
    per_class_total = {i: 0 for i in range(len(class_names))}

    # Process images in batches
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"\tEvaluating...")
    for images, labels in tqdm(dataloader, desc=f"{dataset_name}"):
        with torch.no_grad():
            # Encode images
            image_features = model.encode_image(images)
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
            pred_idx = pred.item()

            per_class_total[label_idx] += 1
            if pred_idx == label_idx:
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

    return {
        "dataset": dataset_name,
        "num_classes": len(class_names),
        "num_images": len(dataset),
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "per_class_accuracy": per_class_accuracy,
        "class_names": class_names,
        "template": template
    }


def evaluate_model(model_name, pretrained, datasets):
    """
    Evaluate a single CLIP model on multiple datasets.

    Args:
        model_name: Model architecture name (e.g., "ViT-B-32).
        pretrained: Pretrained weights name.
        datasets: List of dataset names to evaluate.

    Returns:
        Dictionary with results for each dataset.
    """
    print("\n" + "="*60)
    print(f"Model: {model_name} ({pretrained})")
    print("="*60)

    # Load model
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {num_params:,}")

    # Evaluate on each dataset
    results = {
        "model": model_name,
        "pretrained": pretrained,
        "num_params": num_params,
        "datasets": {}
    }

    for dataset_name in datasets:
        try:
            dataset_results = evaluate_dataset(
                model, preprocess, tokenizer, dataset_name
            )
            results["datasets"][dataset_name] = dataset_results
        except Exception as e:
            print(f"\tError on {dataset_name}: {e}")
            results["datasets"][dataset_name] = {"error": str(e)}

    return results


#
# Visualization
#

def plot_results(all_results, save_path):
    """
    Create a bar chart comparing model performance across datasets.

    Args:
        all_results: List of result dictionaries from eval_model().
        save_path: Optional path to save the figure.
    """
    # Extract data for plotting
    datasets = list(all_results[0]["datasets"].keys())
    models = [f"{r['model']}" for r in all_results]

    # Create accuracy matrix
    accuracy = []
    for result in all_results:
        model_accuracy = []
        for dataset in datasets:
            if "accuracy" in result["datasets"][dataset]:
                model_accuracy.append(result["datasets"][dataset]["accuracy"])
            else:
                model_accuracy.append(0)
        accuracy.append(model_accuracy)

    # Plot
    x = np.arange(len(datasets))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (model_name, model_accuracy) in enumerate(zip(models, accuracy)):
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, model_accuracy, width, label=model_name)
        # Add value labels on bars
        for bar, acc in zip(bars, model_accuracy):
            ax.annotate(f'{acc:.1f}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    ax.set_ylabel('Zero-Shot Accuracy (%)')
    ax.set_title('CLIP Zero-Shot Classification Accuracy by Model and Dataset')
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha='right')
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {save_path}")

    plt.show()



def print_summary_table(all_results):
    """
    Print a summary table of results.
    """
    print("\n" + "="*80)
    print("SUMMARY: Zero-Shot Accuracy (%)")
    print("="*80)

    # Header
    datasets = list(all_results[0]["datasets"].keys())
    header = f"{'Model:':<25} | " + " | ".join(f"{d:>10}" for d in datasets)
    print(header)
    print("-" * len(header))

    # Rows
    for result in all_results:
        model_name = result["model"]
        accuracy = []
        for dataset in datasets:
            if "accuracy" in result["datasets"][dataset]:
                accuracy.append(
                    f"{result['datasets'][dataset]['accuracy']:>10.2f}")
            else:
                accuracy.append(f"{'ERROR':>10}")
            row = f"{model_name:<25} | " + " | ".join(accuracy)
            print(row)

    print("="*80)


def plot_confusion_matrix(predictions, labels, class_names, model_name,
                          dataset_name, save_path=None):
    """
    Create and save a confusion matrix visualization.
    """
    cm = confusion_matrix(labels, predictions)

    # Normalize by row to show percentages
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized,
                                  display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='.2f')

    plt.title(f'Confusion Matrix: {model_name} on {dataset_name}')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Confision matrix saved to: {save_path}")

    plt.show()


def per_class_table(all_results, dataset_name):
    """
    Build a per-class accuracy table for a given dataset.

    Returns a DataFrame with class names as rows and models as columns.
    """
    rows = {}
    for result in all_results:
        model = result["model"]
        dataset = result["datasets"].get(dataset_name, {})
        per_class = dataset.get("per_class_accuracy", {})
        for class_name, accuracy in per_class.items():
            if class_name not in rows:
                rows[class_name] = {}
            rows[class_name][model] = accuracy

    return pd.DataFrame(rows).T.sort_index()


def all_per_class_tables(all_results):
    dataset_names = all_results[0]["datasets"].keys()
    return {name: per_class_table(all_results, name) for name in dataset_names}


def load_results(json_path):
    """
    This function is for when you run your code for three hours
    and realize there's a bug at the very end that makes it crash.
    """
    with open(json_path, "r") as f:
        return json.load(f)

#
# Main loop
#

def evaluate_dataset_with_confusion(model, preprocess, tokenizer,
                                    dataset_name, batch_size=64):
    """
    Same as evaluate_dataset but also returns predictions and labels for
    confusion matrix.
    """
    # Load dataset
    dataset, class_names, template = get_dataset(dataset_name, preprocess)
    print(f"\n Dataset: {dataset_name}")
    print(f"Classes: {len(class_names)}")
    print(f"Images: {len(dataset)}")

    # Encode text prompts
    text_features = encode_text_prompts(model, tokenizer, class_names, template)

    # Store all predictions and labels
    all_predictions = []
    all_labels = []

    # Process images
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    print(f"Evaluating...")
    for images, labels in tqdm(dataloader, desc=f"{dataset_name}"):
        with torch.no_grad():
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T)
            predictions = similarity.argmax(dim=-1)

        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    # Compute accuracy
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    accuracy = 100.0 * (all_predictions == all_labels).sum() / len(all_labels)

    print(f"Accuracy: {accuracy:.2f}%")

    return all_predictions, all_labels, class_names, accuracy


def run_confusion_analysis(dataset_name="eurosat"):
    """
    Generate confusion matrix for EuroSAT with ViT-L-14
    """
    model_name = "ViT-L-14"
    pretrained = "laion2b_s32b_b82k"

    print(f"Loading {model_name}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained)
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    # Get predictions
    predictions, labels, class_names, accuracy = evaluate_dataset_with_confusion(
        model, preprocess, tokenizer, dataset_name
    )

    # Plot confusion matrix
    plot_confusion_matrix(
        predictions, labels, class_names,
        model_name, dataset_name,
        save_path=f"./results/confusion_matrix_{dataset_name}_{model_name}.png"
    )

    return predictions, labels, class_names


def runOpenClip():
    """
    Run the full evaluation.
    """
    print("CLIP Zero-Shot Evaluation")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Datasets: {DATASETS}")
    print(f"Models: {[m[0] for m in MODELS]}")

    # Create results directory
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Evaluate each model
    all_results = []
    for model_name, pretrained in MODELS:
        results = evaluate_model(model_name, pretrained, DATASETS)
        all_results.append(results)

    # Print summary
    print_summary_table(all_results)

    # Save results to JSON
    results_path = os.path.join(RESULTS_DIR, "clip_evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # all_results = load_results("./results/clip_evaluation_results.json")
    # tables = all_per_class_tables(all_results)
    # for dataset_name, table in tables.items():
    #     print(f"\n=== {dataset_name} ===")
    #     print(table)

    # Plot results
    plot_path = os.path.join(RESULTS_DIR, "clip_evaluation_plot.png")
    plot_results(all_results, save_path=plot_path)

    return all_results


# Collect raw (un-preprocessed) images for display.
# Each torchvision dataset stores raw images differently, so we read
# them directly rather than toggling the transform on and off.
def get_raw_image(dataset, idx):
    """
    Return a clean PIL RGB image for display, bypassing the dataset
    transform entirely. Storage format varies by dataset:
      - CIFAR-10/100:  dataset.data[idx]         → numpy (H, W, 3)
      - OxfordIIITPet: dataset._images[idx]      → file path
      - Flowers102:    dataset._image_files[idx] → file path
      - EuroSAT:       dataset.imgs[idx][0]      → file path
    """
    if hasattr(dataset, "data"):
        # CIFAR-10 / CIFAR-100
        return Image.fromarray(dataset.data[idx]).convert("RGB")
    elif hasattr(dataset, "_images"):
        # OxfordIIITPet
        return Image.open(dataset._images[idx]).convert("RGB")
    elif hasattr(dataset, "_image_files"):
        # Flowers102
        return Image.open(dataset._image_files[idx]).convert("RGB")
    elif hasattr(dataset, "imgs"):
        # EuroSAT
        return Image.open(dataset.imgs[idx][0]).convert("RGB")
    else:
        raise ValueError(f"Unknown dataset type: {type(dataset)}")


def plot_clip_predictions(model, preprocess, tokenizer, dataset, class_names,
                          template, dataset_label, n_images=8, seed=42,
                          save_path=None):
    """
    Plot CLIP zero-shot predictions in the OpenAI style:
    image on the left, horizontal probability bar chart on the right.

    Each row shows one image. The top predicted label is shown above
    as "predicted_class (XX.X%) Ranked N out of M Labels".
    The correct label bar is blue; all others are gray.

    Args:
       model:         CLIP model (eval mode).
       preprocess:    Image preprocessing transform.
       tokenizer:     CLIP tokenizer.
       dataset:       Torchvision dataset (returns image_tensor, label_index).
       class_names:   List of human-readable class name strings.
       template:      Prompt template string, e.g. "a photo of a {}".
       dataset_label: Display name shown as the figure title (e.g. "Oxford-IIIT Pets").
       n_images:      How many sample images to show.
       seed:          Random seed for reproducible sampling.
       save_path:     If provided, saves the figure to this path.
    """
    # Sample n_images random indices from the dataset.
    rng = np.random.default_rng() # seed
    indices = rng.choice(len(dataset), size=n_images, replace=False)

    # Build text embeddings for all classes
    # Shape: (num_classes, embed_dim)
    text_features = encode_text_prompts(model, tokenizer, class_names, template)
    num_classes = len(class_names)

    pil_images = []
    true_labels = []
    for idx in indices:
        img = get_raw_image(dataset, idx)
        _, label = dataset[idx]  # Use dataset to get the correct label
        pil_images.append(img)
        true_labels.append(label)

    # Run CLIP on each sampled image
    # Preprocess each PIL image and stack into a batch tensor
    image_tensors = torch.stack([preprocess(img) for img in pil_images])

    with torch.no_grad():
        # Encode images -> normalize -> (n_images, embed_dim)
        image_features = model.encode_image(image_tensors)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Cosine similarity scaled to logits -> softmax -> probabilities
        # Shape: (n_images, num_classes)
        logits = 100.0 * image_features @ text_features.T
        probs = logits.softmax(dim=-1).cpu().numpy()

    # Build the figure
    # Each image gets one row: left=image, right=bar chart
    fig, axes = plt.subplots(
        nrows=n_images,
        ncols=2,
        figsize=(11, 2.6 * n_images),
        gridspec_kw={"width_ratios": [1, 2.5]} # Image column narrower
    )

    # Dark background to match OpenAI page style
    fig.patch.set_facecolor("#1a1a1a")
    fig.suptitle(dataset_label, fontsize=16, fontweight="bold",
                 color="white", y=1.01)

    for row, (pil_img, true_label, prob_row) in enumerate(
        zip(pil_images, true_labels, probs)):

        # Top-5 class indices by probability (highest first)
        top5_indices = np.argsort(prob_row)[::-1][:5]
        top5_probs = prob_row[top5_indices]
        top5_names = [class_names[i] for i in top5_indices]
        top5_prompts = [template.format(n) for n in top5_names]

        # Rank of the true class among ALL classes (1 = best)
        sorted_all = np.argsort(prob_row)[::-1]
        true_rank = int(np.where(sorted_all == true_label)[0][0]) + 1

        predicted_class = class_names[top5_indices[0]]
        predicted_prob = top5_probs[0] * 100  # convert to percentage
        correct = (top5_indices[0] == true_label)

        # Left panel: image
        ax_img = axes[row, 0]
        ax_img.imshow(pil_img)
        ax_img.axis("off")
        ax_img.set_facecolor("#1a1a1a")

        # True label watermark on the image (white text, top-left)
        ax_img.text(
            0.04, 0.96,
            class_names[true_label],
            transform=ax_img.transAxes,
            fontsize=9, fontweight="bold",
            color="white", va="top",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5)
        )

        # Right panel: horizontal bar chart
        ax_bar = axes[row, 1]
        ax_bar.set_facecolor("#2b2b2b")

        # Draw bars bottom-to-top so rank-1 appears at the top
        y_pos = np.arange(5)[::-1] # [4, 3, 2, 1, 0] → rank1 at top
        bar_colors = []
        for idx in top5_indices:
            if idx == true_label:
                bar_colors.append("#29af58") # Blue = correct
            else:
                bar_colors.append("#C33A3A") # Gray = wrong

        bars = ax_bar.barh(
            y_pos, top5_probs * 100,
            color=bar_colors,
            height=0.6,
            edgecolor="none"
        )

        # Y-tick labels = full prompt text
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(
            top5_prompts,
            fontsize=9, color="white"
        )

        # Bold the class name within the prompt label
        # (matplotlib doesn't support inline bold, so we annotate separately)
        ax_bar.set_xlim(0, max(top5_probs * 100) * 1.25 + 2)
        ax_bar.tick_params(axis="x", colors="#888888", labelsize=8)
        ax_bar.tick_params(axis="y", length=0)
        ax_bar.spines[["top", "right", "left", "bottom"]].set_visible(False)
        ax_bar.xaxis.set_visible(False)

        # Checkmark / X prefix on each bar label
        # for i, (bar, class_idx) in enumerate(zip(bars, top5_indices)):
        #     marker = "✓" if class_idx == true_label else "✗"
        #     marker_color = "#29af58" if class_idx == true_label else "#C33A3A"
        #     ax_bar.text(
        #         -0.5, y_pos[i],
        #         marker,
        #         ha="right", va="center",
        #         fontsize=10, color=marker_color,
        #         transform=ax_bar.get_yaxis_transform()
        #     )

        # Header text above each bar row:
        # "PredictedClass (XX.X%)  Ranked N out of M labels"
        header_color = "#29af58" if correct else "#C33A3A"
        ax_bar.set_title(
            f"{predicted_class} ({predicted_prob:.1f}%)   "
            f"Ranked {true_rank} out of {num_classes} labels",
            fontsize=10, color=header_color,
            loc="left", pad=6, fontweight="bold"
        )
        ax_bar.set_facecolor("#2b2b2b")

    plt.tight_layout(h_pad=1.2)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"Saved: {save_path}")

    # plt.show()




def plot_clip_predictions_demo(dataset_name="cifar10", model_name="ViT-L-14",
                               pretrained="laion2b_s32b_b82k", n_images=5):
    """
    Quick demo: loads ViT-B-32 (small/fast model) and runs plot_clip_predictions
    on a sample from the requested dataset.

    Use dataset_name="cifar10" for the fastest test (~2 min CPU).

    Args: dataset_name: One of cifar10, cifar100, flowers102, pets, eurosat.
    n_images:           Number of sample images to display
    """
    # Display names matching OpenAI page style
    display_names = {
        "cifar10": "CIFAR-10",
        "cifar100": "CIFAR-100",
        "flowers102": "Flowers-102",
        "pets": "Oxford-IIIT Pets",
        "eurosat": "EuroSAT"
    }

    print(f"Loading {model_name}...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    # Load dataset (with preprocessing for model).
    dataset, class_names, template = get_dataset(dataset_name, preprocess)
    save_path = f"./results/predictions_{dataset_name}_{model_name}.png"

    plot_clip_predictions(
        model=model,
        preprocess=preprocess,
        tokenizer=tokenizer,
        dataset=dataset,
        class_names=class_names,
        template=template,
        dataset_label=display_names.get(dataset_name, dataset_name),
        n_images=n_images,
        save_path=save_path,
    )


if __name__ == "__main__":
    # plot_clip_predictions_demo(dataset_name="cifar10")
    # plot_clip_predictions_demo(dataset_name="cifar100")
    # plot_clip_predictions_demo(dataset_name="flowers102")
    # plot_clip_predictions_demo(dataset_name="pets")
    # plot_clip_predictions_demo(dataset_name="eurosat")

    # results = runOpenClip()
    # run_confusion_analysis("eurosat")
    run_confusion_analysis("pets")
    run_confusion_analysis("flowers102")