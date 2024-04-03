from os.path import join
import torch
from torchvision import transforms
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152
from PIL import Image

from utils import ASSET_PATH
from data.dataloader import Data
from representations.representation import Representation

clip_image = join(ASSET_PATH, "images", "representations", "CLIP.png")
citation = """
```latex
    @InProceedings{He_2016_CVPR,
        author = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
        title = {Deep Residual Learning for Image Recognition},
        booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
        month = {June},
        year = {2016}
    } 
```
"""

SHORT_DESCRIPTION = \
    f"""
---

Residual Networks (ResNets) are a revolutionary neural network architecture designed to facilitate the training of much deeper models by introducing "residual blocks" with skip connections. These connections allow the network to learn residual functions by adding the input of a block to its output, which helps combat the vanishing gradient problem commonly encountered in deep networks. This design enables the construction of networks that are significantly deeper than previous models, improving performance across a wide range of tasks, particularly in image recognition and classification. ResNets have set new benchmarks in accuracy for various computer vision applications and have been adapted and extended in numerous ways, influencing the development of many subsequent deep learning architectures.

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
Residual Networks (ResNets) are a type of deep neural network architecture designed to enable the training of much deeper networks by addressing the vanishing gradient problem. Introduced by Kaiming He et al. in their 2015 paper, "Deep Residual Learning for Image Recognition," ResNets have become a foundational model for many computer vision tasks and beyond, significantly improving performance on benchmarks for image classification, object detection, and more.

### Key Features of ResNets:

1. **Residual Blocks**: The core idea behind ResNets is the introduction of "residual blocks" with skip connections. These blocks allow the input to a layer to be added to its output, effectively enabling the network to learn a residual mapping. This approach makes it easier to optimize the network and alleviates the vanishing gradient problem, as gradients can flow directly through the skip connections.
2. **Skip Connections**: Skip connections, or shortcuts, bypass one or more layers and perform identity mapping, adding the input of the block to its output. These connections are crucial for enabling the training of deep networks by preserving the gradient signal.
3. **Deep Architectures**: With the introduction of residual blocks, ResNets can be built with a much larger number of layers—models with over a hundred layers have been successfully trained. Deep ResNets have shown remarkable performance improvements over shallower architectures, setting new records in various tasks.
4. **Stack of Residual Blocks**: A ResNet is composed of several stacked residual blocks. Each block typically contains a few convolutional layers, and the depth of the network can be adjusted by changing the number of these blocks.
5. **Batch Normalization**: ResNets also make extensive use of batch normalization, which normalizes the inputs to layers within the network. This normalization helps in stabilizing and accelerating the training process.

### Variants and Improvements:

Since the original introduction of ResNets, numerous variants and improvements have been proposed, including ResNet-v2, Wide ResNets (WRNs), and ResNeXt. These models introduce modifications such as pre-activation residual units, increased width (more channels per layer), and grouped convolutions, further enhancing the model's performance and efficiency.

### Applications:

ResNets have been applied to a wide range of tasks beyond image classification, including object detection, semantic segmentation, and even non-vision tasks such as speech recognition and natural language processing. Their ability to be trained deeply while maintaining performance has made them a go-to architecture for many deep learning challenges.

In summary, ResNets represent a significant breakthrough in deep learning, enabling the training of networks that are much deeper than was previously feasible. Their design principles, particularly the use of residual blocks and skip connections, have influenced many subsequent neural network architectures.

&nbsp;

The original code can be found here:
[https://github.com/KaimingHe/deep-residual-networks](https://github.com/KaimingHe/deep-residual-networks)


---

## Paper

Title:

> **Deep Residual Learning for Image Recognition**

&nbsp;

Abstract:

>Deeper neural networks are more difficult to train. We present a residual learning framework to ease the training of networks that are substantially deeper than those used previously. We explicitly reformulate the layers as learning residual functions with reference to the layer inputs, instead of learning unreferenced functions. We provide comprehensive empirical evidence showing that these residual networks are easier to optimize, and can gain accuracy from considerably increased depth. On the ImageNet dataset we evaluate residual nets with a depth of up to 152 layers---8x deeper than VGG nets but still having lower complexity. An ensemble of these residual nets achieves 3.57% error on the ImageNet test set. This result won the 1st place on the ILSVRC 2015 classification task. We also present analysis on CIFAR-10 with 100 and 1000 layers. The depth of representations is of central importance for many visual recognition tasks. Solely due to our extremely deep representations, we obtain a 28% relative improvement on the COCO object detection dataset. Deep residual nets are foundations of our submissions to ILSVRC & COCO 2015 competitions, where we also won the 1st places on the tasks of ImageNet detection, ImageNet localization, COCO detection, and COCO segmentation.

&nbsp;

If you use ResNet, please cite the following paper:

{citation}
"""


class ResNet(Representation):
    def __init__(self, root):
        super().__init__(root=root)

        self.parameter.update({
            "weights": "resnet18",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        })
        self.parameter_choices.update({
            "weights": [
                "resnet18", "resnet34", "resnet50", "resnet101", "resnet152"
            ],
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
        weights = self.parameter["weights"]
        if weights == "resnet18":
            model = resnet18(pretrained=True)
        elif weights == "resnet34":
            model = resnet34(pretrained=True)
        elif weights == "resnet50":
            model = resnet50(pretrained=True)
        elif weights == "resnet101":
            model = resnet101(pretrained=True)
        elif weights == "resnet152":
            model = resnet152(pretrained=True)
        else:
            raise ValueError(f"Unknown weights: {weights}")
        model = torch.nn.Sequential(*(list(model.children())[:-1]))
        model.eval()

        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # move the input and model to GPU for speed if available
        device = "cpu"
        if torch.cuda.is_available() and self.parameter["device"]:
            device = "cuda"
        model.to(device)

        with torch.no_grad():
            features = list()
            for i in range(len(data)):
                print(f"\r({i}/{len(data)})", end="")
                img = Image.fromarray(data.get_image(i))

                image = preprocess(img).unsqueeze(0).to(
                    device=device)
                image_features = model(image)
                features.append(image_features)
            features = torch.concatenate(features, dim=0).cpu().numpy()

        return features
