from os.path import join
import torch
import clip
from PIL import Image

from utils import ASSET_PATH
from data.dataloader import Data
from representations.representation import Representation

clip_image = join(ASSET_PATH, "images", "representations", "CLIP.png")
citation = """
```latex
    @article{DBLP:journals/corr/abs-2103-00020,
        author       = {Alec Radford and
                        Jong Wook Kim and
                        Chris Hallacy and
                        Aditya Ramesh and
                        Gabriel Goh and
                        Sandhini Agarwal and
                        Girish Sastry and
                        Amanda Askell and
                        Pamela Mishkin and
                        Jack Clark and
                        Gretchen Krueger and
                        Ilya Sutskever},
        title        = {Learning Transferable Visual Models From Natural Language Supervision},
        journal      = {CoRR},
        volume       = {abs/2103.00020},
        year         = {2021},
        url          = {https://arxiv.org/abs/2103.00020},
        eprinttype    = {arXiv},
        eprint       = {2103.00020},
        timestamp    = {Thu, 04 Mar 2021 17:00:40 +0100},
        biburl       = {https://dblp.org/rec/journals/corr/abs-2103-00020.bib},
        bibsource    = {dblp computer science bibliography, https://dblp.org}
    }
```
"""

SHORT_DESCRIPTION = \
    f"""
---

CLIP (Contrastive Language–Image Pretraining) is a method developed by OpenAI for creating image embeddings. It involves training two neural networks, one for images and one for text, using a large-scale dataset of image-text pairs. Through a contrastive learning approach, CLIP learns to align the embeddings of images and their corresponding textual descriptions, enabling it to understand a wide range of visual concepts. This method is notable for its zero-shot learning capabilities, allowing it to perform various vision tasks without task-specific training. CLIP's adaptability and scalability make it a powerful tool for diverse applications in the field of computer vision.

---

<|{clip_image}|image|class_name=centered_full|>

---

## How to use


The models are pre-trained, ready for immediate application in inference tasks. This pre-training ensures a high level of accuracy and efficiency right out of the box, eliminating the need for additional training time or resources. However, to optimize the model's performance for your specific use case, there are a couple of key parameters that you should configure:

- **Weights**: The 'weights' parameter determines the specific model weights to be used. The default setting is 'ViT-B/32', which is well-suited for a broad range of applications. These weights have been meticulously trained and are integral to the model's ability to process and analyze images.
- **Device**: The 'device' parameter specifies the hardware that the model will run on. By default, the model will utilize 'cuda' if it's available, offering accelerated computing performance. In the absence of 'cuda', it will automatically switch to 'cpu'. This flexibility ensures that the model can be deployed in various environments without compatibility issues.

By adjusting these parameters, you can tailor the model to your specific requirements,
ensuring optimal performance during inference. Remember, no additional training is required; 
simply configure these settings, safe the model and the model is ready to deliver 
its full potential.

    """

DESCRIPTION = \
    f"""
---

## Introduction
CLIP, which stands for Contrastive Language–Image Pretraining, is an innovative method developed by OpenAI for encoding images into embeddings that can be effectively used for a wide range of vision tasks. Here's a detailed description of the CLIP method:

1. **Dual-Modality Design**: CLIP is based on a dual-modality approach. It simultaneously trains two neural networks: one for processing images and another for processing text. The image encoder transforms an image into a high-dimensional vector, or an embedding, while the text encoder does the same for a given piece of text.
2. **Contrastive Learning Framework**: The core of CLIP’s training process lies in its contrastive learning framework. This framework involves presenting the model with pairs of related images and text (such as an image and its corresponding descriptive caption). The goal is for the model to learn to match the image embeddings and the text embeddings when they are related.
3. **Large-Scale Dataset**: One of the key strengths of CLIP is its training on a large-scale dataset. It uses a vast and diverse dataset of images and their associated textual descriptions scraped from the internet. This enables the model to learn a wide variety of visual concepts and associations.
4. **Zero-Shot Learning Capabilities**: A remarkable feature of CLIP is its ability to perform well on tasks it wasn't explicitly trained on, known as zero-shot learning. After training, CLIP can be applied to various tasks like object recognition, classification, or even geolocation of images, without the need for additional fine-tuning specific to these tasks.
5. **Embedding Space Alignment**: During training, CLIP aims to align the embedding spaces of images and texts. This means that semantically similar images and texts have closer embeddings in the high-dimensional space. For example, the embedding for an image of a 'dog' and the text "a photo of a dog" would be close to each other in the embedding space.
6. **Scalability and Adaptability**: CLIP's design allows it to scale with more data and compute, potentially improving its performance and adaptability to various domains and applications. This scalability is crucial for adapting to the ever-growing diversity of visual content on the internet.

In summary, CLIP encodes images into embeddings by training on a vast dataset of image-text pairs, using a contrastive learning approach to align the embeddings of semantically related images and texts. This results in a versatile model capable of understanding a wide range of visual concepts and applicable to various vision tasks without task-specific training.

&nbsp;

The original code is maintained by OpenAI and can be found here:
[https://github.com/openai/CLIP](https://github.com/openai/CLIP)


---

## Paper

<|{clip_image}|image|class_name=centered_full|>

Title:

> **Learning Transferable Visual Models From Natural Language Supervision**

&nbsp;


Abstract:


>State-of-the-art computer vision systems are trained to predict a fixed set of predetermined object categories. This restricted form of supervision limits their generality and usability since additional labeled data is needed to specify any other visual concept. Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision. We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet. After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks. We study the performance of this approach by benchmarking on over 30 different existing computer vision datasets, spanning tasks such as OCR, action recognition in videos, geo-localization, and many types of fine-grained object classification. The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training. For instance, we match the accuracy of the original ResNet-50 on ImageNet zero-shot without needing to use any of the 1.28 million training examples it was trained on. We release our code and pre-trained model weights at https://github.com/OpenAI/CLIP

&nbsp;

If you use CLIP, please cite the following paper:

{citation}
"""


class CLIP(Representation):
    def __init__(self, root):
        super().__init__(root=root)

        self.parameter.update({
            "weights": "ViT-B/32",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        })
        self.parameter_choices.update({
            "weights": clip.available_models(),
            "device": ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"],
        })
        self.__str__ = SHORT_DESCRIPTION
        self.__doc__ = DESCRIPTION

    def _train(self, data: Data, **kwargs):
        return self.parameter

    def _load_model(self, model) -> bool:
        if type(model) is not dict:
            return False
        if model.keys() != self.parameter.keys():
            return False
        self.parameter.update(model)
        return True

    def _inference(self, data: Data, **kwargs):
        model, preprocess = clip.load(
            self.parameter["weights"], device=self.parameter["device"])

        with torch.no_grad():
            features = list()
            for i in range(len(data)):
                print(f"\r({i}/{len(data)})", end="")
                img = Image.fromarray(data.get_image(i))
                image = preprocess(img).unsqueeze(0).to(
                    self.parameter["device"])
                image_features = model.encode_image(image)
                features.append(image_features)
            features = torch.concatenate(features, dim=0).cpu().numpy()

        return features
