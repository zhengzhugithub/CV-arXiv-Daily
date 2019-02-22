# CV-arXiv-Daily

**分享计算机视觉每天的arXiv文章，主要集中在目标检测，单目标跟踪，多目标跟踪，人体行为识别，人体姿态估计与跟踪，行人重识别，模型搜索等。每周周末会将本周的Archive起来**

[2019-01-23~2019-01-26](2019/2019.01.23-2019.01.26.md)

[2019-01-28~2019-02-01](2019/2019.01.28-2019.02.01.md)

[2019-02-04~2019-02-08](2019/2019.02.04-2019.02.08.md)

[2019-02-11~2019-02-15](2019/2019.02.11-2019.02.15.md)

**2019-02-22**

[1] 眨眼检测数据集

论文题目：Towards Real-time Eyeblink Detection in The Wild:Dataset,Theory and Practices

作者：Guilei Hu, Yang Xiao, Zhiguo Cao, Lubin Meng, Zhiwen Fang, Joey Tianyi Zhou

论文链接：https://arxiv.org/abs/1902.07891

摘要: Effective and real-time eyeblink detection is of wide-range applications, such as deception detection, drive fatigue detection, face anti-spoofing, etc. Although numerous of efforts have already been paid, most of them focus on addressing the eyeblink detection problem under the constrained indoor conditions with the relative consistent subject and environment setup. Nevertheless, towards the practical applications eyeblink detection in the wild is more required, and of greater challenges. However, to our knowledge this has not been well studied before. In this paper, we shed the light to this research topic. A labelled eyeblink in the wild dataset (i.e., HUST-LEBW) of 673 eyeblink video samples (i.e., 381 positives, and 292 negatives) is first established by us. These samples are captured from the unconstrained movies, with the dramatic variation on human attribute, human pose, illumination condition, imaging configuration, etc. Then, we formulate eyeblink detection task as a spatial-temporal pattern recognition problem. After locating and tracking human eye using SeetaFace engine and KCF tracker respectively, a modified LSTM model able to capture the multi-scale temporal information is proposed to execute eyeblink verification. A feature extraction approach that reveals appearance and motion characteristics simultaneously is also proposed. The experiments on HUST-LEBW reveal the superiority and efficiency of our approach. It also verifies that, the existing eyeblink detection methods cannot achieve satisfactory performance in the wild.

[2] GAN文章

论文题目：Domain Partitioning Network

作者：Botos Csaba, Adnane Boukhayma, Viveka Kulharia, András Horváth, Philip H. S. Torr

论文链接：https://arxiv.org/abs/1902.08134

摘要: Standard adversarial training involves two agents, namely a generator and a discriminator, playing a mini-max game. However, even if the players converge to an equilibrium, the generator may only recover a part of the target data distribution, in a situation commonly referred to as mode collapse. In this work, we present the Domain Partitioning Network (DoPaNet), a new approach to deal with mode collapse in generative adversarial learning. We employ multiple discriminators, each encouraging the generator to cover a different part of the target distribution. To ensure these parts do not overlap and collapse into the same mode, we add a classifier as a third agent in the game. The classifier decides which discriminator the generator is trained against for each sample. Through experiments on toy examples and real images, we show the merits of DoPaNet in covering the real distribution and its superiority with respect to the competing methods. Besides, we also show that we can control the modes from which samples are generated using DoPaNet.

[3] SLAM开源框架

论文题目：GSLAM: A General SLAM Framework and Benchmark

作者：Yong Zhao, Shibiao Xu, Shuhui Bu, Hongkai Jiang, Pengcheng Han

论文链接：https://arxiv.org/abs/1902.07995

摘要: SLAM technology has recently seen many successes and attracted the attention of high-technological companies. However, how to unify the interface of existing or emerging algorithms, and effectively perform benchmark about the speed, robustness and portability are still problems. In this paper, we propose a novel SLAM platform named GSLAM, which not only provides evaluation functionality, but also supplies useful toolkit for researchers to quickly develop their own SLAM systems. The core contribution of GSLAM is an universal, cross-platform and full open-source SLAM interface for both research and commercial usage, which is aimed to handle interactions with input dataset, SLAM implementation, visualization and applications in an unified framework. Through this platform, users can implement their own functions for better performance with plugin form and further boost the application to practical usage of the SLAM.

**2019-02-21**

[1] adversarial research toolbox using PyTorch

论文题目：advertorch v0.1: An Adversarial Robustness Toolbox based on PyTorch

作者：Gavin Weiguang Ding, Luyu Wang, Xiaomeng Jin

论文链接：https://arxiv.org/abs/1902.07623

代码链接：https://github.com/BorealisAI/advertorch

摘要: advertorch is a toolbox for adversarial robustness research. It contains various implementations for attacks, defenses and robust training methods. advertorch is built on PyTorch (Paszke et al., 2017), and leverages the advantages of the dynamic computational graph to provide concise and efficient reference implementations. The code is licensed under the LGPL license and is open sourced

[2] 实时语义分割文章

论文题目：An efficient solution for semantic segmentation: ShuffleNet V2 with atrous separable convolutions

作者：Sercan Türkmen, Janne Heikkilä

论文链接：https://arxiv.org/abs/1902.07476

摘要: Assigning a label to each pixel in an image, namely semantic segmentation, has been an important task in computer vision, and has applications in autonomous driving, robotic navigation, localization, and scene understanding. Fully convolutional neural networks have proved to be a successful solution for the task over the years but most of the work being done focuses primarily on accuracy. In this paper, we present a computationally efficient approach to semantic segmentation, meanwhile achieving a high mIOU, 70.33% on Cityscapes challenge. The network proposed is capable of running real-time on mobile devices. In addition, we make our code and model weights publicly available.

[3] 自适应感受野网络组件

论文题目：Spatially-Adaptive Filter Units for Compact and Efficient Deep Neural Networks

作者：Domen Tabernik, Matej Kristan, Aleš Leonardis

论文链接：https://arxiv.org/abs/1902.07474

摘要: Convolutional neural networks excel in a number of computer vision tasks. One of their most crucial architectural elements is the effective receptive field size, that has to be manually set to accommodate a specific task. Standard solutions involve large kernels, down/up-sampling and dilated convolutions. These require testing a variety of dilation and down/up-sampling factors and result in non-compact representations and excessive number of parameters. We address this issue by proposing a new convolution filter composed of displaced aggregation units (DAU). DAUs learn spatial displacements and adapt the receptive field sizes of individual convolution filters to a given problem, thus eliminating the need for hand-crafted modifications. DAUs provide a seamless substitution of convolutional filters in existing state-of-the-art architectures, which we demonstrate on AlexNet, ResNet50, ResNet101, DeepLab and SRN-DeblurNet. The benefits of this design are demonstrated on a variety of computer vision tasks and datasets, such as image classification (ILSVRC 2012), semantic segmentation (PASCAL VOC 2011, Cityscape) and blind image de-blurring (GOPRO). Results show that DAUs efficiently allocate parameters resulting in up to four times more compact networks at similar or better performance.

[4] AAAI-19 Action Recognition文章

论文题目：Learning Transferable Self-attentive Representations for Action Recognition in Untrimmed Videos with Weak Supervision

作者：Xiao-Yu Zhang, Haichao Shi, Changsheng Li, Kai Zheng, Xiaobin Zhu, Lixin Duan

论文链接：https://arxiv.org/abs/1902.07370

摘要: Action recognition in videos has attracted a lot of attention in the past decade. In order to learn robust models, previous methods usually assume videos are trimmed as short sequences and require ground-truth annotations of each video frame/sequence, which is quite costly and time-consuming. In this paper, given only video-level annotations, we propose a novel weakly supervised framework to simultaneously locate action frames as well as recognize actions in untrimmed videos. Our proposed framework consists of two major components. First, for action frame localization, we take advantage of the self-attention mechanism to weight each frame, such that the influence of background frames can be effectively eliminated. Second, considering that there are trimmed videos publicly available and also they contain useful information to leverage, we present an additional module to transfer the knowledge from trimmed videos for improving the classification performance in untrimmed ones. Extensive experiments are conducted on two benchmark datasets (i.e., THUMOS14 and ActivityNet1.3), and experimental results clearly corroborate the efficacy of our method.

[5] AAAI-19 Motion Prediction文章

论文题目：Human Motion Prediction via Learning Local Structure Representations and Temporal Dependencies

作者：Xiao Guo, Jongmoo Choi

论文链接：https://arxiv.org/abs/1902.07367

摘要: Human motion prediction from motion capture data is a classical problem in the computer vision, and conventional methods take the holistic human body as input. These methods ignore the fact that, in various human activities, different body components (limbs and the torso) have distinctive characteristics in terms of the moving pattern. In this paper, we argue local representations on different body components should be learned separately and, based on such idea, propose a network, Skeleton Network (SkelNet), for long-term human motion prediction. Specifically, at each time-step, local structure representations of input (human body) are obtained via SkelNet's branches of component-specific layers, then the shared layer uses local spatial representations to predict the future human pose. Our SkelNet is the first to use local structure representations for predicting the human motion. Then, for short-term human motion prediction, we propose the second network, named as Skeleton Temporal Network (Skel-TNet). Skel-TNet consists of three components: SkelNet and a Recurrent Neural Network, they have advantages in learning spatial and temporal dependencies for predicting human motion, respectively; a feed-forward network that outputs the final estimation. Our methods achieve promising results on the Human3.6M dataset and the CMU motion capture dataset.

[6] 针对小目标检测的数据增强方式

论文题目：Augmentation for small object detection

作者：Mate Kisantal, Zbigniew Wojna, Jakub Murawski, Jacek Naruniec, Kyunghyun Cho

论文链接：https://arxiv.org/abs/1902.07296

摘要: In recent years, object detection has experienced impressive progress. Despite these improvements, there is still a significant gap in the performance between the detection of small and large objects. We analyze the current state-of-the-art model, Mask-RCNN, on a challenging dataset, MS COCO. We show that the overlap between small ground-truth objects and the predicted anchors is much lower than the expected IoU threshold. We conjecture this is due to two factors; (1) only a few images are containing small objects, and (2) small objects do not appear enough even within each image containing them. We thus propose to oversample those images with small objects and augment each of those images by copy-pasting small objects many times. It allows us to trade off the quality of the detector on large objects with that on small objects. We evaluate different pasting augmentation strategies, and ultimately, we achieve 9.7\% relative improvement on the instance segmentation and 7.1\% on the object detection of small objects, compared to the current state of the art method on MS COCO.

**2019-02-20**

[1] 分割图像标注工具FreeLabel

论文题目：FreeLabel: A Publicly Available Annotation Tool based on Freehand Traces

作者：Philipe A. Dias, Zhou Shen, Amy Tabb, Henry Medeiros

论文链接：https://arxiv.org/abs/1902.06806

摘要: Large-scale annotation of image segmentation datasets is often prohibitively expensive, as it usually requires a huge number of worker hours to obtain high-quality results. Abundant and reliable data has been, however, crucial for the advances on image understanding tasks achieved by deep learning models. In this paper, we introduce FreeLabel, an intuitive open-source web interface that allows users to obtain high-quality segmentation masks with just a few freehand scribbles, in a matter of seconds. The efficacy of FreeLabel is quantitatively demonstrated by experimental results on the PASCAL dataset as well as on a dataset from the agricultural domain. Designed to benefit the computer vision community, FreeLabel can be used for both crowdsourced or private annotation and has a modular structure that can be easily adapted for any image dataset.


[2] 一分半钟训练AlexNet

论文题目：Optimizing Network Performance for Distributed DNN Training on GPU Clusters: ImageNet/AlexNet Training in 1.5 Minutes

作者：Peng Sun, Wansen Feng, Ruobing Han, Shengen Yan, Yonggang Wen

论文链接：https://arxiv.org/abs/1902.06855

摘要: It is important to scale out deep neural network (DNN) training for reducing model training time. The high communication overhead is one of the major performance bottlenecks for distributed DNN training across multiple GPUs. Our investigations have shown that popular open-source DNN systems could only achieve 2.5 speedup ratio on 64 GPUs connected by 56 Gbps network. To address this problem, we propose a communication backend named GradientFlow for distributed DNN training, and employ a set of network optimization techniques. First, we integrate ring-based allreduce, mixed-precision training, and computation/communication overlap into GradientFlow. Second, we propose lazy allreduce to improve network throughput by fusing multiple communication operations into a single one, and design coarse-grained sparse communication to reduce network traffic by only transmitting important gradient chunks. When training ImageNet/AlexNet on 512 GPUs, our approach achieves 410.2 speedup ratio and completes 95-epoch training in 1.5 minutes, which outperforms existing approaches.

[3] WIDER Face and Pedestrian Challenge 2018官方报告

论文题目：WIDER Face and Pedestrian Challenge 2018: Methods and Results

作者：Chen Change Loy等

论文链接：https://arxiv.org/abs/1902.06854

摘要: This paper presents a review of the 2018 WIDER Challenge on Face and Pedestrian. The challenge focuses on the problem of precise localization of human faces and bodies, and accurate association of identities. It comprises of three tracks: (i) WIDER Face which aims at soliciting new approaches to advance the state-of-the-art in face detection, (ii) WIDER Pedestrian which aims to find effective and efficient approaches to address the problem of pedestrian detection in unconstrained environments, and (iii) WIDER Person Search which presents an exciting challenge of searching persons across 192 movies. In total, 73 teams made valid submissions to the challenge tracks. We summarize the winning solutions for all three tracks. and present discussions on open problems and potential research directions in these topics.

[4] 人体部件检测文章

论文题目：Detector-in-Detector: Multi-Level Analysis for Human-Parts

作者：Xiaojie Li, Lu Yang, Qing Song, Fuqiang Zhou

论文链接：https://arxiv.org/abs/1902.07017

摘要: Vision-based person, hand or face detection approaches have achieved incredible success in recent years with the development of deep convolutional neural network (CNN). In this paper, we take the inherent correlation between the body and body parts into account and propose a new framework to boost up the detection performance of the multi-level objects. In particular, we adopt a region-based object detection structure with two carefully designed detectors to separately pay attention to the human body and body parts in a coarse-to-fine manner, which we call Detector-in-Detector network (DID-Net). The first detector is designed to detect human body, hand, and face. The second detector, based on the body detection results of the first detector, mainly focus on the detection of small hand and face inside each body. The framework is trained in an end-to-end way by optimizing a multi-task loss. Due to the lack of human body, face and hand detection dataset, we have collected and labeled a new large dataset named Human-Parts with 14,962 images and 106,879 annotations. Experiments show that our method can achieve excellent performance on Human-Parts.

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














