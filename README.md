# CV-arXiv-Daily

**分享计算机视觉每天的arXiv文章，主要集中在目标检测，单目标跟踪，多目标跟踪，人体行为识别，人体姿态估计与跟踪，行人重识别，模型搜索等。每周周末会将本周的Archive起来**

[2019-01-23~2019-01-26](2019/2019.01.23-2019.01.26.md)

**2019-01-28**

[1] Google的自监督表征学习文章

Revisiting Self-Supervised Visual Representation Learning

论文链接：https://arxiv.org/abs/1901.09005

代码地址：https://github.com/google/revisiting-self-supervised

摘要: Unsupervised visual representation learning remains a largely unsolved problem in computer vision research. Among a big body of recently proposed approaches for unsupervised learning of visual representations, a class of self-supervised techniques achieves superior performance on many challenging benchmarks. A large number of the pretext tasks for self-supervised learning have been studied, but other important aspects, such as the choice of convolutional neural networks (CNN), has not received equal attention. Therefore, we revisit numerous previously proposed self-supervised models, conduct a thorough large scale study and, as a result, uncover multiple crucial insights. We challenge a number of common practices in selfsupervised visual representation learning and observe that standard recipes for CNN design do not always translate to self-supervised representation learning. As part of our study, we drastically boost the performance of previously proposed techniques and outperform previously published state-of-the-art results by a large margin.

[2] LCLR 2019 GAN文章

Diversity-Sensitive Conditional Generative Adversarial Networks

论文链接：https://arxiv.org/abs/1901.09024

摘要: We propose a simple yet highly effective method that addresses the mode-collapse problem in the Conditional Generative Adversarial Network (cGAN). Although conditional distributions are multi-modal (i.e., having many modes) in practice, most cGAN approaches tend to learn an overly simplified distribution where an input is always mapped to a single output regardless of variations in latent code. To address such issue, we propose to explicitly regularize the generator to produce diverse outputs depending on latent codes. The proposed regularization is simple, general, and can be easily integrated into most conditional GAN objectives. Additionally, explicit regularization on generator allows our method to control a balance between visual quality and diversity. We demonstrate the effectiveness of our method on three conditional generation tasks: image-to-image translation, image inpainting, and future video prediction. We show that simple addition of our regularization to existing models leads to surprisingly diverse generations, substantially outperforming the previous approaches for multi-modal conditional generation specifically designed in each individual task.

[3] 上交卢策吾老师的Q-learning for斗地主 文章

Combinational Q-Learning for Dou Di Zhu

论文链接：https://arxiv.org/abs/1901.08925

代码地址：https://github.com/qq456cvb/doudizhu-C

摘要: Deep reinforcement learning (DRL) has gained a lot of attention in recent years, and has been proven to be able to play Atari games and Go at or above human levels. However, those games are assumed to have a small fixed number of actions and could be trained with a simple CNN network. In this paper, we study a special class of Asian popular card games called Dou Di Zhu, in which two adversarial groups of agents must consider numerous card combinations at each time step, leading to huge number of actions. We propose a novel method to handle combinatorial actions, which we call combinational Q-learning (CQL). We employ a two-stage network to reduce action space and also leverage order-invariant max-pooling operations to extract relationships between primitive actions. Results show that our method prevails over state-of-the art methods like naive Q-learning and A3C. We develop an easy-to-use card game environments and train all agents adversarially from sractch, with only knowledge of game rules and verify that our agents are comparative to humans. Our code to reproduce all reported results will be available online.

[4] WACV2019 3D点云 文章

Dense 3D Point Cloud Reconstruction Using a Deep Pyramid Network

论文链接：https://arxiv.org/abs/1901.08906

摘要: Reconstructing a high-resolution 3D model of an object is a challenging task in computer vision. Designing scalable and light-weight architectures is crucial while addressing this problem. Existing point-cloud based reconstruction approaches directly predict the entire point cloud in a single stage. Although this technique can handle low-resolution point clouds, it is not a viable solution for generating dense, high-resolution outputs. In this work, we introduce DensePCR, a deep pyramidal network for point cloud reconstruction that hierarchically predicts point clouds of increasing resolution. Towards this end, we propose an architecture that first predicts a low-resolution point cloud, and then hierarchically increases the resolution by aggregating local and global point features to deform a grid. Our method generates point clouds that are accurate, uniform and dense. Through extensive quantitative and qualitative evaluation on synthetic and real datasets, we demonstrate that DensePCR outperforms the existing state-of-the-art point cloud reconstruction works, while also providing a light-weight and scalable architecture for predicting high-resolution outputs.

[5] Multi-Target Multi-Camera Tracking 文章

Multiple Hypothesis Tracking Algorithm for Multi-Target Multi-Camera Tracking with Disjoint Views

论文链接：https://arxiv.org/abs/1901.08787

摘要: In this study, a multiple hypothesis tracking (MHT) algorithm for multi-target multi-camera tracking (MCT) with disjoint views is proposed. Our method forms track-hypothesis trees, and each branch of them represents a multi-camera track of a target that may move within a camera as well as move across cameras. Furthermore, multi-target tracking within a camera is performed simultaneously with the tree formation by manipulating a status of each track hypothesis. Each status represents three different stages of a multi-camera track: tracking, searching, and end-of-track. The tracking status means targets are tracked by a single camera tracker. In the searching status, the disappeared targets are examined if they reappear in other cameras. The end-of-track status does the target exited the camera network due to its lengthy invisibility. These three status assists MHT to form the track-hypothesis trees for multi-camera tracking. Furthermore, they present a gating technique for eliminating of unlikely observation-to-track association. In the experiments, they evaluate the proposed method using two datasets, DukeMTMC and NLPR-MCT, which demonstrates that the proposed method outperforms the state-of-the-art method in terms of improvement of the accuracy. In addition, they show that the proposed method can operate in real-time and online.


[6] One-Class CNN 文章

One-Class Convolutional Neural Network

论文链接：https://arxiv.org/abs/1901.08688

代码地址：github.com/otkupjnoz/oc-cnn

摘要: We present a novel Convolutional Neural Network (CNN) based approach for one class classification. The idea is to use a zero centered Gaussian noise in the latent space as the pseudo-negative class and train the network using the cross-entropy loss to learn a good representation as well as the decision boundary for the given class. A key feature of the proposed approach is that any pre-trained CNN can be used as the base network for one class classification. The proposed One Class CNN (OC-CNN) is evaluated on the UMDAA-02 Face, Abnormality-1001, FounderType-200 datasets. These datasets are related to a variety of one class application problems such as user authentication, abnormality detection and novelty detection. Extensive experiments demonstrate that the proposed method achieves significant improvements over the recent state-of-the-art methods. The source code is available at : github.com/otkupjnoz/oc-cnn.


[7] In Defense of the Triplet Loss 文章

In Defense of the Triplet Loss for Visual Recognition

论文链接：https://arxiv.org/abs/1901.08616

摘要: We employ triplet loss as a space embedding regularizer to boost classification performance. Standard architectures, like ResNet and DesneNet, are extended to support both losses with minimal hyper-parameter tuning. This promotes generality while fine-tuning pretrained networks. Triplet loss is a powerful surrogate for recently proposed embedding regularizers. Yet, it is avoided for large batch-size requirement and high computational cost. Through our experiments, we re-assess these assumptions. 
During inference, our network supports both classification and embedding tasks without any computational overhead. Quantitative evaluation highlights how our approach compares favorably to the existing state of the art on multiple fine-grained recognition datasets. Further evaluation on an imbalanced video dataset achieves significant improvement (>7%). Beyond boosting efficiency, triplet loss brings retrieval and interpretability to classification models.






