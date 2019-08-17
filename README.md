# CV-arXiv-Daily

**分享计算机视觉每天的arXiv文章，主要集中在目标检测，单目标跟踪，多目标跟踪，人体行为识别，人体姿态估计与跟踪，行人重识别，模型搜索等。每周周末会将本周的Archive起来**

[2019-01-23~2019-01-26](2019/2019.01.23-2019.01.26.md)

[2019-01-28~2019-02-01](2019/2019.01.28-2019.02.01.md)

[2019-02-04~2019-02-08](2019/2019.02.04-2019.02.08.md)

[2019-02-11~2019-02-15](2019/2019.02.11-2019.02.15.md)

[2019-02-18~2019-02-22](2019/2019.02.18-2019.02.22.md)

[2019-02-25~2019-03-01](2019/2019.02.25-2019.03.01.md)

**2019-08-16**

[1] FastPose人体姿态估计与跟踪算法

论文题目：FastPose: Towards Real-time Pose Estimation and Tracking via Scale-normalized Multi-task Networks

作者：Jiabin Zhang, Zheng Zhu, Wei Zou, Peng Li, Yanwei Li, Hu Su, Guan Huang

论文链接：https://arxiv.org/abs/1908.05593

摘要: Both accuracy and efficiency are significant for pose estimation and tracking in videos. State-of-the-art performance is dominated by two-stages top-down methods. Despite the leading results, these methods are impractical for real-world applications due to their separated architectures and complicated calculation. This paper addresses the task of articulated multi-person pose estimation and tracking towards real-time speed. An end-to-end multi-task network (MTN) is designed to perform human detection, pose estimation, and person re-identification (Re-ID) tasks simultaneously. To alleviate the performance bottleneck caused by scale variation problem, a paradigm which exploits scale-normalized image and feature pyramids (SIFP) is proposed to boost both performance and speed. Given the results of MTN, we adopt an occlusion-aware Re-ID feature strategy in the pose tracking module, where pose information is utilized to infer the occlusion state to make better use of Re-ID feature. In experiments, we demonstrate that the pose estimation and tracking performance improves steadily utilizing SIFP through different backbones. Using ResNet-18 and ResNet-50 as backbones, the overall pose tracking framework achieves competitive performance with 29.4 FPS and 12.2 FPS, respectively. Additionally, occlusion-aware Re-ID feature decreases the identification switches by 37% in the pose tracking process.


[2] R3Det检测算法

论文题目：R3Det: Refined Single-Stage Detector with Feature Refinement for Rotating Object

作者：Xue Yang, Qingqing Liu, Junchi Yan, Ang Li

论文链接：https://arxiv.org/abs/1908.05612

摘要: Rotation detection is a challenging task due to the difficulties of locating the multi-angle objects and separating them accurately and quickly from the background. Though considerable progress has been made, there still exist challenges for rotating objects with large aspect ratio, dense distribution and category extremely imbalance. In this paper, we propose an end-to-end refined single-stage rotation detector for fast and accurate positioning objects. Considering the shortcoming of feature misalignment in the current refined single-stage detector, we design a feature refinement module to improve detection performance, which is especially effective in the long tail data set. The key idea of feature refinement module is to re-encode the position information of the current refined bounding box to the corresponding feature points through feature interpolation to realize feature reconstruction and alignment. Extensive experiments on two remote sensing public datasets DOTA, HRSC2016 as well as scene text data ICDAR2015 show the state-of-the-art accuracy and speed of our detector. Source code and the models will be made public available upon the publish of the paper.

[3] SAST文本检测算法

论文题目：A Single-Shot Arbitrarily-Shaped Text Detector based on Context Attended Multi-Task Learning

作者：Pengfei Wang, Chengquan Zhang, Fei Qi, Zuming Huang, Mengyi En, Junyu Han, Jingtuo Liu, Errui Ding, Guangming Shi

论文链接：https://arxiv.org/abs/1908.05498

摘要: Detecting scene text of arbitrary shapes has been a challenging task over the past years. In this paper, we propose a novel segmentation-based text detector, namely SAST, which employs a context attended multi-task learning framework based on a Fully Convolutional Network (FCN) to learn various geometric properties for the reconstruction of polygonal representation of text regions. Taking sequential characteristics of text into consideration, a Context Attention Block is introduced to capture long-range dependencies of pixel information to obtain a more reliable segmentation. In post-processing, a Point-to-Quad assignment method is proposed to cluster pixels into text instances by integrating both high-level object knowledge and low-level pixel information in a single shot. Moreover, the polygonal representation of arbitrarily-shaped text can be extracted with the proposed geometric properties much more effectively. Experiments on several benchmarks, including ICDAR2015, ICDAR2017-MLT, SCUT-CTW1500, and Total-Text, demonstrate that SAST achieves better or comparable performance in terms of accuracy. Furthermore, the proposed algorithm runs at 27.63 FPS on SCUT-CTW1500 with a Hmean of 81.0% on a single NVIDIA Titan Xp graphics card, surpassing most of the existing segmentation-based methods.



**2019-04-24**

[1] Re-ID文章

论文题目：Interpretable and Generalizable Deep Image Matching with Adaptive Convolutions

作者：Shengcai Liao and Ling Shao

论文链接：https://arxiv.org/abs/1904.10424

摘要: For image matching tasks, like face recognition and person re-identification, existing deep networks often focus on representation learning. However, without domain adaptation or transfer learning, the learned model is fixed as is, which is not adaptable to handle various unseen scenarios. In this paper, beyond representation learning, we consider how to formulate image matching directly in deep feature maps. We treat image matching as finding local correspondences in feature maps, and construct adaptive convolution kernels on the fly to achieve local matching. In this way, the matching process and result is interpretable, and this explicit matching is more generalizable than representation features to unseen scenarios, such as unknown misalignments, pose or viewpoint changes. To facilitate end-to-end training of such an image matching architecture, we further build a class memory module to cache feature maps of the most recent samples of each class, so as to compute image matching losses for metric learning. The proposed method is preliminarily validated on the person re-identification task. Through direct cross-dataset evaluation without further transfer learning, it achieves better results than many transfer learning methods. Besides, a model-free temporal cooccurrence based score weighting method is proposed, which improves the performance to a further extent, resulting in state-of-the-art results in cross-dataset evaluation.

[2] 基于CornerNet和CenterNet的Visual Tracking文章

论文题目：Siamese Attentional Keypoint Network for High Performance Visual Tracking

作者：Peng Gao, Yipeng Ma, Ruyue Yuan, Liyi Xiao, Fei Wang

论文链接：https://arxiv.org/abs/1904.10128

摘要: In this paper, we investigate impacts of three main aspects of visual tracking, i.e., the backbone network, the attentional mechanism and the detection component, and propose a Siamese Attentional Keypoint Network, dubbed SATIN, to achieve efficient tracking and accurate localization. Firstly, a new Siamese lightweight hourglass network is specifically designed for visual tracking. It takes advantage of the benefits of the repeated bottom-up and top-down inference to capture more global and local contextual information at multiple scales. Secondly, a novel cross-attentional module is utilized to leverage both channel-wise and spatial intermediate attentional information, which enhance both discriminative and localization capabilities of feature maps. Thirdly, a keypoints detection approach is invented to track any target object by detecting the top-left corner point, the centroid point and the bottom-right corner point of its bounding box. To the best of our knowledge, we are the first to propose this approach. Therefore, our SATIN tracker not only has a strong capability to learn more effective object representations, but also computational and memory storage efficiency, either during the training or testing stage. Without bells and whistles, experimental results demonstrate that our approach achieves state-of-the-art performance on several recent benchmark datasets, at speeds far exceeding the frame-rate requirement.


**2019-04-23**

[1] NAS for face

论文题目：Neural Architecture Search for Deep Face Recognition

作者：Ning Zhu, Xiaolong Bai

论文链接：https://arxiv.org/abs/1904.09523

摘要: By the widespread popularity of electronic devices, the emergence of biometric technology has brought significant convenience to user authentication compared with the traditional password and mode unlocking. Among many biological characteristics, the face is a universal and irreplaceable feature that does not need too much cooperation and can significantly improve the user's experience at the same time. Face recognition is one of the main functions of electronic equipment propaganda. Hence it's virtually worth researching in computer vision. Previous work in this field has focused on two directions: converting loss function to improve recognition accuracy in traditional deep convolution neural networks (Resnet); combining the latest loss function with the lightweight system (MobileNet) to reduce network size at the minimal expense of accuracy. But none of these has changed the network structure. With the development of AutoML, neural architecture search (NAS) has shown excellent performance in the benchmark of image classification. In this paper, we integrate NAS technology into face recognition to customize a more suitable network. We quote the framework of neural architecture search which trains child and controller network alternately. At the same time, we mutate NAS by incorporating evaluation latency into rewards of reinforcement learning and utilize policy gradient algorithm to search the architecture automatically with the most classical cross-entropy loss. The network architectures we searched out have got state-of-the-art accuracy in the large-scale face dataset, which achieves 98.77% top-1 in MS-Celeb-1M and 99.89% in LFW with relatively small network size. To the best of our knowledge, this proposal is the first attempt to use NAS to solve the problem of Deep Face Recognition and achieve the best results in this domain.

[2] 一种新的Normalization方法SW

论文题目：Switchable Whitening for Deep Representation Learning

作者：Xingang Pan, Xiaohang Zhan, Jianping Shi, Xiaoou Tang, Ping Luo

论文链接：https://arxiv.org/abs/1904.09739

摘要: Normalization methods are essential components in convolutional neural networks (CNNs). They either standardize or whiten data using statistics estimated in predefined sets of pixels. Unlike existing works that design normalization techniques for specific tasks, we propose Switchable Whitening (SW), which provides a general form unifying different whitening methods as well as standardization methods. SW learns to switch among these operations in an end-to-end manner. It has several advantages. First, SW adaptively selects appropriate whitening or standardization statistics for different tasks (see Fig.1), making it well suited for a wide range of tasks without manual design. Second, by integrating benefits of different normalizers, SW shows consistent improvements over its counterparts in various challenging benchmarks. Third, SW serves as a useful tool for understanding the characteristics of whitening and standardization techniques. We show that SW outperforms other alternatives on image classification (CIFAR-10/100, ImageNet), semantic segmentation (ADE20K, Cityscapes), domain adaptation (GTA5, Cityscapes), and image style transfer (COCO). For example, without bells and whistles, we achieve state-of-the-art performance with 45.33% mIoU on the ADE20K dataset. Code and models will be released.



**2019-04-03**

[1] FAIR Kaiming等人的NAS文章

论文题目：Exploring Randomly Wired Neural Networks for Image Recognition

作者：Saining Xie, Alexander Kirillov, Ross Girshick, Kaiming He

论文链接：https://arxiv.org/abs/1904.01569

摘要: Neural networks for image recognition have evolved through extensive manual design from simple chain-like models to structures with multiple wiring paths. The success of ResNets and DenseNets is due in large part to their innovative wiring plans. Now, neural architecture search (NAS) studies are exploring the joint optimization of wiring and operation types, however, the space of possible wirings is constrained and still driven by manual design despite being searched. In this paper, we explore a more diverse set of connectivity patterns through the lens of randomly wired neural networks. To do this, we first define the concept of a stochastic network generator that encapsulates the entire network generation process. Encapsulation provides a unified view of NAS and randomly wired networks. Then, we use three classical random graph models to generate randomly wired graphs for networks. The results are surprising: several variants of these random generators yield network instances that have competitive accuracy on the ImageNet benchmark. These results suggest that new efforts focusing on designing better network generators may lead to new breakthroughs by exploring less constrained search spaces with more room for novel design.

[2] 目标检测文章

论文题目：FCOS: Fully Convolutional One-Stage Object Detection

作者：Zhi Tian, Chunhua Shen, Hao Chen, Tong He

论文链接：https://arxiv.org/abs/1904.01355

摘要: We propose a fully convolutional one-stage object detector (FCOS) to solve object detection in a per-pixel prediction fashion, analogue to semantic segmentation. Almost all state-of-the-art object detectors such as RetinaNet, SSD, YOLOv3, and Faster R-CNN rely on pre-defined anchor boxes. In contrast, our proposed detector FCOS is anchor-box free, as well as proposal free. By eliminating the pre-defined set of anchor boxes, FCOS completely avoids the complicated computation related to anchor boxes such as calculating overlapping during training and significantly reduces the training memory footprint. More importantly, we also avoid all hyper-parameters related to anchor boxes, which are often very sensitive to the final detection performance. With the only post-processing non-maximum suppression (NMS), our detector FCOS outperforms previous anchor-based one-stage detectors with the advantage of being much simpler. For the first time, we demonstrate a much simpler and flexible detection framework achieving improved detection accuracy. We hope that the proposed FCOS framework can serve as a simple and strong alternative for many other instance-level tasks.

[3] Res2Net

论文题目：Res2Net: A New Multi-scale Backbone Architecture

作者：Shang-Hua Gao, Ming-Ming Cheng, Kai Zhao, Xin-Yu Zhang, Ming-Hsuan Yang, Philip Torr

论文链接：https://arxiv.org/abs/1904.01169

摘要: Representing features at multiple scales is of great importance for numerous vision tasks. Recent advances in backbone convolutional neural networks (CNNs) continually demonstrate stronger multi-scale representation ability, leading to consistent performance gains on a wide range of applications. However, most existing methods represent the multi-scale features in a layer-wise manner. In this paper, we propose a novel building block for CNNs, namely Res2Net, by constructing hierarchical residual-like connections within one single residual block. The Res2Net represents multi-scale features at a granular level and increases the range of receptive fields for each network layer. The proposed Res2Net block can be plugged into the state-of-the-art backbone CNN models, e.g., ResNet, ResNeXt, and DLA. We evaluate the Res2Net block on all these models and demonstrate consistent performance gains over baseline models on widely-used datasets, e.g., CIFAR-100 and ImageNet. Further ablation studies and experimental results on representative computer vision tasks, i.e., object detection, class activation mapping, and salient object detection, further verify the superiority of the Res2Net over the state-of-the-art baseline methods. The source code and trained models will be made publicly available.

**2019-03-29**

[1] Kaiming 新文

论文题目：TensorMask: A Foundation for Dense Object Segmentation

作者：Xinlei Chen, Ross Girshick, Kaiming He, Piotr Dollár

论文链接：https://arxiv.org/abs/1903.12174

摘要: Sliding-window object detectors that generate bounding-box object predictions over a dense, regular grid have advanced rapidly and proven popular. In contrast, modern instance segmentation approaches are dominated by methods that first detect object bounding boxes, and then crop and segment these regions, as popularized by Mask R-CNN. In this work, we investigate the paradigm of dense sliding-window instance segmentation, which is surprisingly under-explored. Our core observation is that this task is fundamentally different than other dense prediction tasks such as semantic segmentation or bounding-box object detection, as the output at every spatial location is itself a geometric structure with its own spatial dimensions. To formalize this, we treat dense instance segmentation as a prediction task over 4D tensors and present a general framework called TensorMask that explicitly captures this geometry and enables novel operators on 4D tensors. We demonstrate that the tensor view leads to large gains over baselines that ignore this structure, and leads to results comparable to Mask R-CNN. These promising results suggest that TensorMask can serve as a foundation for novel advances in dense mask prediction and a more complete understanding of the task. Code will be made available.

[2] 商汤Text Detector文章

论文题目：Pyramid Mask Text Detector

作者：Jingchao Liu, Xuebo Liu, Jie Sheng, Ding Liang, Xin Li, Qingjie Liu

论文链接：https://arxiv.org/abs/1903.11800

摘要: Scene text detection, an essential step of scene text recognition system, is to locate text instances in natural scene images automatically. Some recent attempts benefiting from Mask R-CNN formulate scene text detection task as an instance segmentation problem and achieve remarkable performance. In this paper, we present a new Mask R-CNN based framework named Pyramid Mask Text Detector (PMTD) to handle the scene text detection. Instead of binary text mask generated by the existing Mask R-CNN based methods, our PMTD performs pixel-level regression under the guidance of location-aware supervision, yielding a more informative soft text mask for each text instance. As for the generation of text boxes, PMTD reinterprets the obtained 2D soft mask into 3D space and introduces a novel plane clustering algorithm to derive the optimal text box on the basis of 3D shape. Experiments on standard datasets demonstrate that the proposed PMTD brings consistent and noticeable gain and clearly outperforms state-of-the-art methods. Specifically, it achieves an F-measure of 80.13% on ICDAR 2017 MLT dataset.

[3] 自动化所Semantic Segmentation文章

论文题目：FastFCN: Rethinking Dilated Convolution in the Backbone for Semantic Segmentation

作者：Huikai Wu, Junge Zhang, Kaiqi Huang, Kongming Liang, Yizhou Yu

论文链接：https://arxiv.org/abs/1903.11816

摘要: Modern approaches for semantic segmentation usually employ dilated convolutions in the backbone to extract high-resolution feature maps, which brings heavy computation complexity and memory footprint. To replace the time and memory consuming dilated convolutions, we propose a novel joint upsampling module named Joint Pyramid Upsampling (JPU) by formulating the task of extracting high-resolution feature maps into a joint upsampling problem. With the proposed JPU, our method reduces the computation complexity by more than three times without performance loss. Experiments show that JPU is superior to other upsampling modules, which can be plugged into many existing approaches to reduce computation complexity and improve performance. By replacing dilated convolutions with the proposed JPU module, our method achieves the state-of-the-art performance in Pascal Context dataset (mIoU of 53.13%) and ADE20K dataset (final score of 0.5584) while running 3 times faster.

[4]商汤detection文章

论文题目：Feature Intertwiner for Object Detection

作者：Hongyang Li, Bo Dai, Shaoshuai Shi, Wanli Ouyang, Xiaogang Wang

论文链接：https://arxiv.org/abs/1903.11851

摘要: A well-trained model should classify objects with a unanimous score for every category. This requires the high-level semantic features should be as much alike as possible among samples. To achive this, previous works focus on re-designing the loss or proposing new regularization constraints. In this paper, we provide a new perspective. For each category, it is assumed that there are two feature sets: one with reliable information and the other with less reliable source. We argue that the reliable set could guide the feature learning of the less reliable set during training - in spirit of student mimicking teacher behavior and thus pushing towards a more compact class centroid in the feature space. Such a scheme also benefits the reliable set since samples become closer within the same category - implying that it is easier for the classifier to identify. We refer to this mutual learning process as feature intertwiner and embed it into object detection. It is well-known that objects of low resolution are more difficult to detect due to the loss of detailed information during network forward pass (e.g., RoI operation). We thus regard objects of high resolution as the reliable set and objects of low resolution as the less reliable set. Specifically, an intertwiner is designed to minimize the distribution divergence between two sets. The choice of generating an effective feature representation for the reliable set is further investigated, where we introduce the optimal transport (OT) theory into the framework. Samples in the less reliable set are better aligned with aid of OT metric. Incorporated with such a plug-and-play intertwiner, we achieve an evident improvement over previous state-of-the-arts.

[5]搜索网络的Channel Numbers文章

论文题目：Network Slimming by Slimmable Networks: Towards One-Shot Architecture Search for Channel Numbers

作者：Jiahui Yu, Thomas Huang

论文链接：https://arxiv.org/abs/1903.11728

代码链接：https://github.com/JiahuiYu/slimmable_networks

摘要: We study how to set channel numbers in a neural network to achieve better accuracy under constrained resources (e.g., FLOPs, latency, memory footprint or model size). A simple and one-shot solution, named AutoSlim, is presented. Instead of training many network samples and searching with reinforcement learning, we train a single slimmable network to approximate the network accuracy of different channel configurations. We then iteratively evaluate the trained slimmable model and greedily slim the layer with minimal accuracy drop. By this single pass, we can obtain the optimized channel configurations under different resource constraints. We present experiments with MobileNet v1, MobileNet v2, ResNet-50 and RL-searched MNasNet on ImageNet classification. We show significant improvements over their default channel configurations. We also achieve better accuracy than recent channel pruning methods and neural architecture search methods. 
Notably, by setting optimized channel numbers, our AutoSlim-MobileNet-v2 at 305M FLOPs achieves 74.2% top-1 accuracy, 2.4% better than default MobileNet-v2 (301M FLOPs), and even 0.2% better than RL-searched MNasNet (317M FLOPs). Our AutoSlim-ResNet-50 at 570M FLOPs, without depthwise convolutions, achieves 1.3% better accuracy than MobileNet-v1 (569M FLOPs).

[6]旷视detection文章

论文题目：ThunderNet: Towards Real-time Generic Object Detection

作者：Zheng Qin, Zeming Li, Zhaoning Zhang, Yiping Bao, Gang Yu, Yuxing Peng, Jian Sun

论文链接：https://arxiv.org/abs/1903.11752

摘要: Real-time generic object detection on mobile platforms is a crucial but challenging computer vision task. However, previous CNN-based detectors suffer from enormous computational cost, which hinders them from real-time inference in computation-constrained scenarios. In this paper, we investigate the effectiveness of two-stage detectors in real-time generic detection and propose a lightweight two-stage detector named ThunderNet. In the backbone part, we analyze the drawbacks in previous lightweight backbones and present a lightweight backbone designed for object detection. In the detection part, we exploit an extremely efficient RPN and detection head design. To generate more discriminative feature representation, we design two efficient architecture blocks, Context Enhancement Module and Spatial Attention Module. At last, we investigate the balance between the input resolution, the backbone, and the detection head. Compared with lightweight one-stage detectors, ThunderNet achieves superior performance with only 40% of the computational cost on PASCAL VOC and COCO benchmarks. Without bells and whistles, our model runs at 24.1 fps on an ARM-based device. To the best of our knowledge, this is the first real-time detector reported on ARM platforms. Code will be released for paper reproduction.


**2019-03-05**

[1] CVPR2019 Semantic Scene Completion新文

论文题目：RGBD Based Dimensional Decomposition Residual Network for 3D Semantic Scene Completion

作者：Jie Li, Yu Liu, Dong Gong, Qinfeng Shi, Xia Yuan, Chunxia Zhao, Ian Reid

论文链接：https://arxiv.org/abs/1903.00620

摘要: RGB images differentiate from depth images as they carry more details about the color and texture information, which can be utilized as a vital complementary to depth for boosting the performance of 3D semantic scene completion (SSC). SSC is composed of 3D shape completion (SC) and semantic scene labeling while most of the existing methods use depth as the sole input which causes the performance bottleneck. Moreover, the state-of-the-art methods employ 3D CNNs which have cumbersome networks and tremendous parameters. We introduce a light-weight Dimensional Decomposition Residual network (DDR) for 3D dense prediction tasks. The novel factorized convolution layer is effective for reducing the network parameters, and the proposed multi-scale fusion mechanism for depth and color image can improve the completion and segmentation accuracy simultaneously. Our method demonstrates excellent performance on two public datasets. Compared with the latest method SSCNet, we achieve 5.9% gains in SC-IoU and 5.7% gains in SSC-IOU, albeit with only 21% network parameters and 16.6% FLOPs employed compared with that of SSCNet.

[2] CVPR2019 目标检测新文

论文题目：Feature Selective Anchor-Free Module for Single-Shot Object Detection

作者：Chenchen Zhu, Yihui He, Marios Savvides

论文链接：https://arxiv.org/abs/1903.00621

摘要: We motivate and present feature selective anchor-free (FSAF) module, a simple and effective building block for single-shot object detectors. It can be plugged into single-shot detectors with feature pyramid structure. The FSAF module addresses two limitations brought up by the conventional anchor-based detection: 1) heuristic-guided feature selection; 2) overlap-based anchor sampling. The general concept of the FSAF module is online feature selection applied to the training of multi-level anchor-free branches. Specifically, an anchor-free branch is attached to each level of the feature pyramid, allowing box encoding and decoding in the anchor-free manner at an arbitrary level. During training, we dynamically assign each instance to the most suitable feature level. At the time of inference, the FSAF module can work jointly with anchor-based branches by outputting predictions in parallel. We instantiate this concept with simple implementations of anchor-free branches and online feature selection strategy. Experimental results on the COCO detection track show that our FSAF module performs better than anchor-based counterparts while being faster. When working jointly with anchor-based branches, the FSAF module robustly improves the baseline RetinaNet by a large margin under various settings, while introducing nearly free inference overhead. And the resulting best model can achieve a state-of-the-art 44.6% mAP, outperforming all existing single-shot detectors on COCO.

[3] CVPR2019 3D Shape Segmentation新文

论文题目：PartNet: A Recursive Part Decomposition Network for Fine-grained and Hierarchical Shape Segmentation

作者：Fenggen Yu, Kun Liu, Yan Zhang, Chenyang Zhu, Kai Xu

论文链接：https://arxiv.org/abs/1903.00709

摘要: Deep learning approaches to 3D shape segmentation are typically formulated as a multi-class labeling problem. Existing models are trained for a fixed set of labels, which greatly limits their flexibility and adaptivity. We opt for top-down recursive decomposition and develop the first deep learning model for hierarchical segmentation of 3D shapes, based on recursive neural networks. Starting from a full shape represented as a point cloud, our model performs recursive binary decomposition, where the decomposition network at all nodes in the hierarchy share weights. At each node, a node classifier is trained to determine the type (adjacency or symmetry) and stopping criteria of its decomposition. The features extracted in higher level nodes are recursively propagated to lower level ones. Thus, the meaningful decompositions in higher levels provide strong contextual cues constraining the segmentations in lower levels. Meanwhile, to increase the segmentation accuracy at each node, we enhance the recursive contextual feature with the shape feature extracted for the corresponding part. Our method segments a 3D shape in point cloud into an unfixed number of parts, depending on the shape complexity, showing strong generality and flexibility. It achieves the state-of-the-art performance, both for fine-grained and semantic segmentation, on the public benchmark and a new benchmark of fine-grained segmentation proposed in this work. We also demonstrate its application for fine-grained part refinements in image-to-shape reconstruction.

[4] CVPR2019 图像篡改检测新文

论文题目：AIRD: Adversarial Learning Framework for Image Repurposing Detection

作者：Ayush Jaiswal, Yue Wu, Wael AbdAlmageed, Iacopo Masi, Premkumar Natarajan

论文链接：https://arxiv.org/abs/1903.00788

摘要: Image repurposing is a commonly used method for spreading misinformation on social media and online forums, which involves publishing untampered images with modified metadata to create rumors and further propaganda. While manual verification is possible, given vast amounts of verified knowledge available on the internet, the increasing prevalence and ease of this form of semantic manipulation call for the development of robust automatic ways of assessing the semantic integrity of multimedia data. In this paper, we present a novel method for image repurposing detection that is based on the real-world adversarial interplay between a bad actor who repurposes images with counterfeit metadata and a watchdog who verifies the semantic consistency between images and their accompanying metadata, where both players have access to a reference dataset of verified content, which they can use to achieve their goals. The proposed method exhibits state-of-the-art performance on location-identity, subject-identity and painting-artist verification, showing its efficacy across a diverse set of scenarios.

[5] CVPR2019 开源Hand Pose（oral）新文

论文题目：3D Hand Shape and Pose Estimation from a Single RGB Image

作者：Liuhao Ge, Zhou Ren, Yuncheng Li, Zehao Xue, Yingying Wang, Jianfei Cai, Junsong Yuan

论文链接：https://arxiv.org/abs/1903.00812

代码链接：https://github.com/geliuhao/3DHandShapePosefromRGB

摘要: This work addresses a novel and challenging problem of estimating the full 3D hand shape and pose from a single RGB image. Most current methods in 3D hand analysis from monocular RGB images only focus on estimating the 3D locations of hand keypoints, which cannot fully express the 3D shape of hand. In contrast, we propose a Graph Convolutional Neural Network (Graph CNN) based method to reconstruct a full 3D mesh of hand surface that contains richer information of both 3D hand shape and pose. To train networks with full supervision, we create a large-scale synthetic dataset containing both ground truth 3D meshes and 3D poses. When fine-tuning the networks on real-world datasets without 3D ground truth, we propose a weakly-supervised approach by leveraging the depth map as a weak supervision in training. Through extensive evaluations on our proposed new datasets and two public datasets, we show that our proposed method can produce accurate and reasonable 3D hand mesh, and can achieve superior 3D hand pose estimation accuracy when compared with state-of-the-art methods.

[6] CVPR2019 Referring Expression Grounding新文

论文题目：Improving Referring Expression Grounding with Cross-modal Attention-guided Erasing

作者：Xihui Liu, Zihao Wang, Jing Shao, Xiaogang Wang, Hongsheng Li

论文链接：https://arxiv.org/abs/1903.00839

摘要: Referring expression grounding aims at locating certain objects or persons in an image with a referring expression, where the key challenge is to comprehend and align various types of information from visual and textual domain, such as visual attributes, location and interactions with surrounding regions. Although the attention mechanism has been successfully applied for cross-modal alignments, previous attention models focus on only the most dominant features of both modalities, and neglect the fact that there could be multiple comprehensive textual-visual correspondences between images and referring expressions. To tackle this issue, we design a novel cross-modal attention-guided erasing approach, where we discard the most dominant information from either textual or visual domains to generate difficult training samples online, and to drive the model to discover complementary textual-visual correspondences. Extensive experiments demonstrate the effectiveness of our proposed method, which achieves state-of-the-art performance on three referring expression grounding datasets.

[7] CVPR2019 Video Highlight Detection新文

论文题目：Less is More: Learning Highlight Detection from Video Duration

作者：Bo Xiong, Yannis Kalantidis, Deepti Ghadiyaram, Kristen Grauman

论文链接：https://arxiv.org/abs/1903.00859

摘要: Highlight detection has the potential to significantly ease video browsing, but existing methods often suffer from expensive supervision requirements, where human viewers must manually identify highlights in training videos. We propose a scalable unsupervised solution that exploits video duration as an implicit supervision signal. Our key insight is that video segments from shorter user-generated videos are more likely to be highlights than those from longer videos, since users tend to be more selective about the content when capturing shorter videos. Leveraging this insight, we introduce a novel ranking framework that prefers segments from shorter videos, while properly accounting for the inherent noise in the (unlabeled) training data. We use it to train a highlight detector with 10M hashtagged Instagram videos. In experiments on two challenging public video highlight detection benchmarks, our method substantially improves the state-of-the-art for unsupervised highlight detection.

[8] CVPR2019 网络稳健性新文

论文题目：A Kernelized Manifold Mapping to Diminish the Effect of Adversarial Perturbations

作者：Saeid Asgari Taghanaki, Kumar Abhishek, Shekoofeh Azizi, Ghassan Hamarneh

论文链接：https://arxiv.org/abs/1903.01015

摘要: The linear and non-flexible nature of deep convolutional models makes them vulnerable to carefully crafted adversarial perturbations. To tackle this problem, we propose a non-linear radial basis convolutional feature mapping by learning a Mahalanobis-like distance function. Our method then maps the convolutional features onto a linearly well-separated manifold, which prevents small adversarial perturbations from forcing a sample to cross the decision boundary. We test the proposed method on three publicly available image classification and segmentation datasets namely, MNIST, ISBI ISIC 2017 skin lesion segmentation, and NIH Chest X-Ray-14. We evaluate the robustness of our method to different gradient (targeted and untargeted) and non-gradient based attacks and compare it to several non-gradient masking defense strategies. Our results demonstrate that the proposed method can increase the resilience of deep convolutional neural networks to adversarial perturbations without accuracy drop on clean data.

[9] CVPR2019 行为识别新文

论文题目：Collaborative Spatio-temporal Feature Learning for Video Action Recognition

作者：Chao Li, Qiaoyong Zhong, Di Xie, Shiliang Pu

论文链接：https://arxiv.org/abs/1903.01197

摘要: Spatio-temporal feature learning is of central importance for action recognition in videos. Existing deep neural network models either learn spatial and temporal features independently (C2D) or jointly with unconstrained parameters (C3D). In this paper, we propose a novel neural operation which encodes spatio-temporal features collaboratively by imposing a weight-sharing constraint on the learnable parameters. In particular, we perform 2D convolution along three orthogonal views of volumetric video data,which learns spatial appearance and temporal motion cues respectively. By sharing the convolution kernels of different views, spatial and temporal features are collaboratively learned and thus benefit from each other. The complementary features are subsequently fused by a weighted summation whose coefficients are learned end-to-end. Our approach achieves state-of-the-art performance on large-scale benchmarks and won the 1st place in the Moments in Time Challenge 2018. Moreover, based on the learned coefficients of different views, we are able to quantify the contributions of spatial and temporal features. This analysis sheds light on interpretability of the model and may also guide the future design of algorithm for video recognition. 

**2019-03-04**

[1] CVPR2019 开源Mask Scoring R-CNN新文

论文题目：Mask Scoring R-CNN

作者：Zhaojin Huang, Lichao Huang, Yongchao Gong, Chang Huang, Xinggang Wang

论文链接：https://arxiv.org/abs/1903.00241

代码链接：https://github.com/zjhuang22/maskscoring_rcnn

摘要: Letting a deep network be aware of the quality of its own predictions is an interesting yet important problem. In the task of instance segmentation, the confidence of instance classification is used as mask quality score in most instance segmentation frameworks. However, the mask quality, quantified as the IoU between the instance mask and its ground truth, is usually not well correlated with classification score. In this paper, we study this problem and propose Mask Scoring R-CNN which contains a network block to learn the quality of the predicted instance masks. The proposed network block takes the instance feature and the corresponding predicted mask together to regress the mask IoU. The mask scoring strategy calibrates the misalignment between mask quality and mask score, and improves instance segmentation performance by prioritizing more accurate mask predictions during COCO AP evaluation. By extensive evaluations on the COCO dataset, Mask Scoring R-CNN brings consistent and noticeable gain with different models, and outperforms the state-of-the-art Mask R-CNN. We hope our simple and effective approach will provide a new direction for improving instance segmentation.

[2] CVPR2019 3D Point Clouds新文

论文题目：Octree guided CNN with Spherical Kernels for 3D Point Clouds

作者：Huan Lei, Naveed Akhtar, Ajmal Mian

论文链接：https://arxiv.org/abs/1903.00343

摘要: We propose an octree guided neural network architecture and spherical convolutional kernel for machine learning from arbitrary 3D point clouds. The network architecture capitalizes on the sparse nature of irregular point clouds, and hierarchically coarsens the data representation with space partitioning. At the same time, the proposed spherical kernels systematically quantize point neighborhoods to identify local geometric structures in the data, while maintaining the properties of translation-invariance and asymmetry. We specify spherical kernels with the help of network neurons that in turn are associated with spatial locations. We exploit this association to avert dynamic kernel generation during network training that enables efficient learning with high resolution point clouds. The effectiveness of the proposed technique is established on the benchmark tasks of 3D object classification and segmentation, achieving new state-of-the-art on ShapeNet and RueMonge2014 datasets.




