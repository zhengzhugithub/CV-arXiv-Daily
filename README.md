# CV-arXiv-Daily

**分享计算机视觉每天的arXiv文章，主要集中在目标检测，单目标跟踪，多目标跟踪，人体行为识别，人体姿态估计与跟踪，行人重识别，模型搜索等。每周周末会将本周的Archive起来**

[2019-01-23~2019-01-26](2019/2019.01.23-2019.01.26.md)

[2019-01-28~2019-02-01](2019/2019.01.28-2019.02.01.md)

[2019-02-04~2019-02-08](2019/2019.02.04-2019.02.08.md)

[2019-02-11~2019-02-15](2019/2019.02.11-2019.02.15.md)

**2019-02-19**

[1] 遥感图像目标检测

论文题目：R2-CNN: Fast Tiny Object Detection in Large-scale Remote Sensing Images

作者：Jiangmiao Pang, Cong Li, Jianping Shi, Zhihai Xu, Huajun Feng

论文链接：https://arxiv.org/abs/1902.06042

摘要: Recently, convolutional neural network has brought impressive improvements for object detection. However, detecting tiny objects in large-scale remote sensing images still remains challenging. Firstly, the extreme large input size makes existing object detection solutions too slow for practical use. Secondly, the massive and complex backgrounds cause serious false alarms. Moreover, the ultra tiny objects increase the difficulty of accurate detection. To tackle these problems, we propose a unified and self-reinforced network called R2-CNN: Remote sensing Region-based Convolutional Neural Network, composing of backbone Tiny-Net, intermediate global attention block, and final classifier and detector. Tiny-Net is a lightweight residual structure which enables fast and powerful features extraction from inputs. Global attention block is built upon Tiny-Net to inhibit false positives. Classifier is then used to predict the existence of target in each patch, and detector is followed to locate them accurately if available. The classifier and detector are mutually reinforced with end-to-end training, which further speed-up the process and avoid false alarms. Effectiveness of R2-CNN is validated on hundreds of GF-1 images and GF-2 images, which are 18000×18192 pixels, 2.0m resolution, and 27620×29200 pixels, 0.8m resolution respectively. Specifically, we can process a GF-1 image in 29.4s on Titian X just with single thread. According to our knowledge, no previous solution can detect tiny object on such huge remote sensing images gracefully. We believe that it is a significant step towards practical real-time remote sensing systems.

[2] TPAMI弱监督目标检测

论文题目：Min-Entropy Latent Model for Weakly Supervised Object Detection

作者：Fang Wan, Pengxu Wei, Zhenjun Han, Jianbin Jiao, Qixiang Ye

论文链接：https://arxiv.org/abs/1902.06057

摘要: Weakly supervised object detection is a challenging task when provided with image category supervision but required to learn, at the same time, object locations and object detectors. The inconsistency between the weak supervision and learning objectives introduces significant randomness to object locations and ambiguity to detectors. In this paper, a min-entropy latent model (MELM) is proposed for weakly supervised object detection. Min-entropy serves as a model to learn object locations and a metric to measure the randomness of object localization during learning. It aims to principally reduce the variance of learned instances and alleviate the ambiguity of detectors. MELM is decomposed into three components including proposal clique partition, object clique discovery, and object localization. MELM is optimized with a recurrent learning algorithm, which leverages continuation optimization to solve the challenging non-convexity problem. Experiments demonstrate that MELM significantly improves the performance of weakly supervised object detection, weakly supervised object localization, and image classification, against the state-of-the-art approaches.

[3] RES-SE-NET

论文题目：RES-SE-NET: Boosting Performance of Resnets by Enhancing Bridge-connections

作者：Varshaneya V, Balasubramanian S, Darshan Gera

论文链接：https://arxiv.org/abs/1902.06066

摘要: One of the ways to train deep neural networks effectively is to use residual connections. Residual connections can be classified as being either identity connections or bridge-connections with a reshaping convolution. Empirical observations on CIFAR-10 and CIFAR-100 datasets using a baseline Resnet model, with bridge-connections removed, have shown a significant reduction in accuracy. This reduction is due to lack of contribution, in the form of feature maps, by the bridge-connections. Hence bridge-connections are vital for Resnet. However, all feature maps in the bridge-connections are considered to be equally important. In this work, an upgraded architecture "Res-SE-Net" is proposed to further strengthen the contribution from the bridge-connections by quantifying the importance of each feature map and weighting them accordingly using Squeeze-and-Excitation (SE) block. It is demonstrated that Res-SE-Net generalizes much better than Resnet and SE-Resnet on the benchmark CIFAR-10 and CIFAR-100 datasets.

[4] 超分辨率综述

论文题目：Deep Learning for Image Super-resolution: A Survey

作者：Zhihao Wang, Jian Chen, Steven C.H. Hoi

论文链接：https://arxiv.org/abs/1902.06068

摘要: Image Super-Resolution (SR) is an important class of image processing techniques to enhance the resolution of images and videos in computer vision. Recent years have witnessed remarkable progress of image super-resolution using deep learning techniques. In this survey, we aim to give a survey on recent advances of image super-resolution techniques using deep learning approaches in a systematic way. In general, we can roughly group the existing studies of SR techniques into three major categories: supervised SR, unsupervised SR, and domain-specific SR. In addition, we also cover some other important issues, such as publicly available benchmark datasets and performance evaluation metrics. Finally, we conclude this survey by highlighting several future directions and open issues which should be further addressed by the community in the future.

[5] 遥感图像理解数据集

论文题目：BigEarthNet: A Large-Scale Benchmark Archive For Remote Sensing Image Understanding

作者：Gencer Sumbul, Marcela Charfuelan, Begüm Demir, Volker Markl

论文链接：https://arxiv.org/abs/1902.06148

摘要: This paper presents a new large-scale multi-label Sentinel-2 benchmark archive, named BigEarthNet. Our archive consists of 590,326 Sentinel-2 image patches, each of which has 10, 20 and 60 meter image bands associated to the pixel sizes of 120x120, 60x60 and 20x20, respectively. Unlike most of the existing archives, each image patch is annotated by multiple land-cover classes (i.e., multi-labels) that are provided from the CORINE Land Cover database of the year 2018 (CLC 2018). The BigEarthNet is 20 times larger than the existing archives in remote sensing (RS) and thus is much more convenient to be used as a training source in the context of deep learning. This paper first addresses the limitations of the existing archives and then describes properties of our archive. Experimental results obtained in the framework of RS image scene classification problems show that a shallow Convolutional Neural Network (CNN) architecture trained on the BigEarthNet provides very high accuracy compared to a state-of-the-art CNN model pre-trained on the ImageNet (which is a very popular large-scale benchmark archive in computer vision). The BigEarthNet opens up promising directions to advance operational RS applications and research in massive Sentinel-2 image archives.

[6] 无监督学习综述

论文题目：Self-supervised Visual Feature Learning with Deep Neural Networks: A Survey

作者：Longlong Jing, Yingli Tian

论文链接：https://arxiv.org/abs/1902.06162

摘要: Large-scale labeled data are generally required to train deep neural networks in order to obtain better performance in visual feature learning from images or videos for computer vision applications. To avoid extensive cost of collecting and annotating large-scale datasets, as a subset of unsupervised learning methods, self-supervised learning methods are proposed to learn general image and video features from large-scale unlabeled data without using any human-annotated labels. This paper provides an extensive review of deep learning-based self-supervised general visual feature learning methods from images or videos. First, the motivation, general pipeline, and terminologies of this field are described. Then the common deep neural network architectures that used for self-supervised learning are summarized. Next, the main components and evaluation metrics of self-supervised learning methods are reviewed followed by the commonly used image and video datasets and the existing self-supervised visual feature learning methods. Finally, quantitative performance comparisons of the reviewed methods on benchmark datasets are summarized and discussed for both image and video feature learning. At last, this paper is concluded and lists a set of promising future directions for self-supervised visual feature learning.

[7] Image attribute transfer文章

论文题目：Fully-Featured Attribute Transfer

作者：De Xie, Muli Yang, Cheng Deng, Wei Liu, Dacheng Tao

论文链接：https://arxiv.org/abs/1902.06258

摘要: Image attribute transfer aims to change an input image to a target one with expected attributes, which has received significant attention in recent years. However, most of the existing methods lack the ability to de-correlate the target attributes and irrelevant information, i.e., the other attributes and background information, thus often suffering from blurs and artifacts. To address these issues, we propose a novel Attribute Manifold Encoding GAN (AME-GAN) for fully-featured attribute transfer, which can modify and adjust every detail in the images. Specifically, our method divides the input image into image attribute part and image background part on manifolds, which are controlled by attribute latent variables and background latent variables respectively. Through enforcing attribute latent variables to Gaussian distributions and background latent variables to uniform distributions respectively, the attribute transfer procedure becomes controllable and image generation is more photo-realistic. Furthermore, we adopt a conditional multi-scale discriminator to render accurate and high-quality target attribute images. Experimental results on three popular datasets demonstrate the superiority of our proposed method in both performances of the attribute transfer and image generation quality.

[8] 机器人视觉伺服文章

论文题目：DIViS: Domain Invariant Visual Servoing for Collision-Free Goal Reaching

作者：Fereshteh Sadeghi

论文链接：https://arxiv.org/abs/1902.05947

摘要: Robots should understand both semantics and physics to be functional in the real world. While robot platforms provide means for interacting with the physical world they cannot autonomously acquire object-level semantics without needing human. In this paper, we investigate how to minimize human effort and intervention to teach robots perform real world tasks that incorporate semantics. We study this question in the context of visual servoing of mobile robots and propose DIViS, a Domain Invariant policy learning approach for collision free Visual Servoing. DIViS incorporates high level semantics from previously collected static human-labeled datasets and learns collision free servoing entirely in simulation and without any real robot data. However, DIViS can directly be deployed on a real robot and is capable of servoing to the user-specified object categories while avoiding collisions in the real world. DIViS is not constrained to be queried by the final view of goal but rather is robust to servo to image goals taken from initial robot view with high occlusions without this impairing its ability to maintain a collision free path. We show the generalization capability of DIViS on real mobile robots in more than 90 real world test scenarios with various unseen object goals in unstructured environments. DIViS is compared to prior approaches via real world experiments and rigorous tests in simulation. 

**2019-02-18**

[1] Lipschitz GAN

论文题目：Lipschitz Generative Adversarial Nets

作者：Zhiming Zhou, Jiadong Liang, Yuxuan Song, Lantao Yu, Hongwei Wang, Weinan Zhang, Yong Yu, Zhihua Zhang

论文链接：https://arxiv.org/abs/1902.05687

摘要: In this paper we study the convergence of generative adversarial networks (GANs) from the perspective of the informativeness of the gradient of the optimal discriminative function. We show that GANs without restriction on the discriminative function space commonly suffer from the problem that the gradient produced by the discriminator is uninformative to guide the generator. By contrast, Wasserstein GAN (WGAN), where the discriminative function is restricted to 1-Lipschitz, does not suffer from such a gradient uninformativeness problem. We further show in the paper that the model with a compact dual form of Wasserstein distance, where the Lipschitz condition is relaxed, also suffers from this issue. This implies the importance of Lipschitz condition and motivates us to study the general formulation of GANs with Lipschitz constraint, which leads to a new family of GANs that we call Lipschitz GANs (LGANs). We show that LGANs guarantee the existence and uniqueness of the optimal discriminative function as well as the existence of a unique Nash equilibrium. We prove that LGANs are generally capable of eliminating the gradient uninformativeness problem. According to our empirical analysis, LGANs are more stable and generate consistently higher quality samples compared with WGAN.

[2] 医学图像分析综述

论文题目：Going Deep in Medical Image Analysis: Concepts, Methods, Challenges and Future Directions

作者：Fouzia Altaf, Syed M. S. Islam, Naveed Akhtar, Naeem K. Janjua

论文链接：https://arxiv.org/abs/1902.05655

摘要: Medical Image Analysis is currently experiencing a paradigm shift due to Deep Learning. This technology has recently attracted so much interest of the Medical Imaging community that it led to a specialized conference in `Medical Imaging with Deep Learning' in the year 2018. This article surveys the recent developments in this direction, and provides a critical review of the related major aspects. We organize the reviewed literature according to the underlying Pattern Recognition tasks, and further sub-categorize it following a taxonomy based on human anatomy. This article does not assume prior knowledge of Deep Learning and makes a significant contribution in explaining the core Deep Learning concepts to the non-experts in the Medical community. Unique to this study is the Computer Vision/Machine Learning perspective taken on the advances of Deep Learning in Medical Imaging. This enables us to single out `lack of appropriately annotated large-scale datasets' as the core challenge (among other challenges) in this research direction. We draw on the insights from the sister research fields of Computer Vision, Pattern Recognition and Machine Learning etc.; where the techniques of dealing with such challenges have already matured, to provide promising directions for the Medical Imaging community to fully harness Deep Learning in the future.

[3] 街景异常事件检测数据集

论文题目：Street Scene: A new dataset and evaluation protocol for video anomaly detection

作者：Barathkumar Ramachandra, Michael Jones

论文链接：https://arxiv.org/abs/1902.05872

摘要: Progress in video anomaly detection research is currently slowed by small datasets that lack a wide variety of activities as well as flawed evaluation criteria. This paper aims to help move this research effort forward by introducing a large and varied new dataset called Street Scene, as well as two new evaluation criteria that provide a better estimate of how an algorithm will perform in practice. In addition to the new dataset and evaluation criteria, we present two variations of a novel baseline video anomaly detection algorithm and show they are much more accurate on Street Scene than two state-of-the-art algorithms from the literature.

[4] 超分辨率文章

论文题目：Lightweight Feature Fusion Network for Single Image Super-Resolution

作者：Wenming Yang, Wei Wang, Xuechen Zhang, Shuifa Sun, Qingmin Liao

论文链接：https://arxiv.org/abs/1902.05694

摘要: Single image super-resolution(SISR) has witnessed great progress as convolutional neural network(CNN) gets deeper and wider. However, enormous parameters hinder its application to real world problems. In this letter, We propose a lightweight feature fusion network (LFFN) that can fully explore multi-scale contextual information and greatly reduce network parameters while maximizing SISR results. LFFN is built on spindle blocks and a softmax feature fusion module (SFFM). Specifically, a spindle block is composed of a dimension extension unit, a feature exploration unit and a feature refinement unit. The dimension extension layer expands low dimension to high dimension and implicitly learns the feature maps which is suitable for the next unit. The feature exploration unit performs linear and nonlinear feature exploration aimed at different feature maps. The feature refinement layer is used to fuse and refine features. SFFM fuses the features from different modules in a self-adaptive learning manner with softmax function, making full use of hierarchical information with a small amount of parameter cost. Both qualitative and quantitative experiments on benchmark datasets show that LFFN achieves favorable performance against state-of-the-art methods with similar parameters.

[5] VQA文章

论文题目：Cycle-Consistency for Robust Visual Question Answering

作者：Meet Shah, Xinlei Chen, Marcus Rohrbach, Devi Parikh

论文链接：https://arxiv.org/abs/1902.05660

摘要: Despite significant progress in Visual Question Answering over the years, robustness of today's VQA models leave much to be desired. We introduce a new evaluation protocol and associated dataset (VQA-Rephrasings) and show that state-of-the-art VQA models are notoriously brittle to linguistic variations in questions. VQA-Rephrasings contains 3 human-provided rephrasings for 40k questions spanning 40k images from the VQA v2.0 validation dataset. As a step towards improving robustness of VQA models, we propose a model-agnostic framework that exploits cycle consistency. Specifically, we train a model to not only answer a question, but also generate a question conditioned on the answer, such that the answer predicted for the generated question is the same as the ground truth answer to the original question. Without the use of additional annotations, we show that our approach is significantly more robust to linguistic variations than state-of-the-art VQA models, when evaluated on the VQA-Rephrasings dataset. In addition, our approach outperforms state-of-the-art approaches on the standard VQA and Visual Question Generation tasks on the challenging VQA v2.0 dataset.














