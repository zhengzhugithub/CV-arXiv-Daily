# CV-arXiv-Daily

**分享计算机视觉每天的arXiv文章，主要集中在目标检测，单目标跟踪，多目标跟踪，人体行为识别，人体姿态估计与跟踪，行人重识别，模型搜索等。每周周末会将本周的Archive起来**

[2019-01-23~2019-01-26](2019/2019.01.23-2019.01.26.md)

[2019-01-28~2019-02-01](2019/2019.01.28-2019.02.01.md)

[2019-02-04~2019-02-08](2019/2019.02.04-2019.02.08.md)

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

**2019-02-15**

[1] DeepLab后续之DeeperLab

论文题目：DeeperLab: Single-Shot Image Parser

作者：Tien-Ju Yang, Maxwell D. Collins, Yukun Zhu, Jyh-Jing Hwang, Ting Liu, Xiao Zhang, Vivienne Sze, George Papandreou, Liang-Chieh Chen

论文链接：https://arxiv.org/abs/1902.05093

摘要: We present a single-shot, bottom-up approach for whole image parsing. Whole image parsing, also known as Panoptic Segmentation, generalizes the tasks of semantic segmentation for 'stuff' classes and instance segmentation for 'thing' classes, assigning both semantic and instance labels to every pixel in an image. Recent approaches to whole image parsing typically employ separate standalone modules for the constituent semantic and instance segmentation tasks and require multiple passes of inference. Instead, the proposed DeeperLab image parser performs whole image parsing with a significantly simpler, fully convolutional approach that jointly addresses the semantic and instance segmentation tasks in a single-shot manner, resulting in a streamlined system that better lends itself to fast processing. For quantitative evaluation, we use both the instance-based Panoptic Quality (PQ) metric and the proposed region-based Parsing Covering (PC) metric, which better captures the image parsing quality on 'stuff' classes and larger object instances. We report experimental results on the challenging Mapillary Vistas dataset, in which our single model achieves 31.95% (val) / 31.6% PQ (test) and 55.26% PC (val) with 3 frames per second (fps) on GPU or near real-time speed (22.6 fps on GPU) with reduced accuracy.

[2] Crowd Counting文章

论文题目：Improving Dense Crowd Counting Convolutional Neural Networks using Inverse k-Nearest Neighbor Maps and Multiscale Upsampling

作者：Greg Olmschenk, Hao Tang, Zhigang Zhu

论文链接：https://arxiv.org/abs/1902.05379

摘要: Gatherings of thousands to millions of people occur frequently for an enormous variety of events, and automated counting of these high density crowds is used for safety, management, and measuring significance of these events. In this work, we show that the regularly accepted labeling scheme of crowd density maps for training deep neural networks is less effective than our alternative inverse k-nearest neighbor (ikNN) maps, even when used directly in existing state-of-the-art network structures. We also provide a new network architecture MUD-ikNN, which uses multi-scale upsampling via transposed convolutions to take full advantage of the provided ikNN labeling. This upsampling combined with the ikNN maps further outperforms the existing state-of-the-art methods. The full label comparison emphasizes the importance of the labeling scheme, with the ikNN labeling being particularly effective. We demonstrate the accuracy of our MUD-ikNN network and the ikNN labeling scheme on a variety of datasets.

[3] Multispectral Pedestrian Detection文章

论文题目：Box-level Segmentation Supervised Deep Neural Networks for Accurate and Real-time Multispectral Pedestrian Detection

作者：Yanpeng Cao, Dayan Guan, Yulun Wu, Jiangxin Yang, Yanlong Cao, Michael Ying Yang

论文链接：https://arxiv.org/abs/1902.05291

摘要: Effective fusion of complementary information captured by multi-modal sensors (visible and infrared cameras) enables robust pedestrian detection under various surveillance situations (e.g. daytime and nighttime). In this paper, we present a novel box-level segmentation supervised learning framework for accurate and real-time multispectral pedestrian detection by incorporating features extracted in visible and infrared channels. Specifically, our method takes pairs of aligned visible and infrared images with easily obtained bounding box annotations as input and estimates accurate prediction maps to highlight the existence of pedestrians. It offers two major advantages over the existing anchor box based multispectral detection methods. Firstly, it overcomes the hyperparameter setting problem occurred during the training phase of anchor box based detectors and can obtain more accurate detection results, especially for small and occluded pedestrian instances. Secondly, it is capable of generating accurate detection results using small-size input images, leading to improvement of computational efficiency for real-time autonomous driving applications. Experimental results on KAIST multispectral dataset show that our proposed method outperforms state-of-the-art approaches in terms of both accuracy and speed.

[4] Temporal Action Localization文章

论文题目：Exploring Frame Segmentation Networks for Temporal Action Localization

作者：Ke Yang, Xiaolong Shen, Peng Qiao, Shijie Li, Dongsheng Li, Yong Dou

论文链接：https://arxiv.org/abs/1902.05488

摘要: Temporal action localization is an important task of computer vision. Though many methods have been proposed, it still remains an open question how to predict the temporal location of action segments precisely. Most state-of-the-art works train action classifiers on video segments pre-determined by action proposal. However, recent work found that a desirable model should move beyond segment-level and make dense predictions at a fine granularity in time to determine precise temporal boundaries. In this paper, we propose a Frame Segmentation Network (FSN) that places a temporal CNN on top of the 2D spatial CNNs. Spatial CNNs are responsible for abstracting semantics in spatial dimension while temporal CNN is responsible for introducing temporal context information and performing dense predictions. The proposed FSN can make dense predictions at frame-level for a video clip using both spatial and temporal context information. FSN is trained in an end-to-end manner, so the model can be optimized in spatial and temporal domain jointly. We also adapt FSN to use it in weakly supervised scenario (WFSN), where only video level labels are provided when training. Experiment results on public dataset show that FSN achieves superior performance in both frame-level action localization and temporal action localization.

**2019-02-14**

[1] Set-based Face Recognition文章

论文题目：Multi-Prototype Networks for Unconstrained Set-based Face Recognition

作者：Jian Zhao, Jianshu Li, Xiaoguang Tu, Fang Zhao, Yuan Xin, Junliang Xing, Hengzhu Liu, Shuicheng Yan, Jiashi Feng

论文链接：https://arxiv.org/abs/1902.04755

摘要: In this paper, we study the challenging unconstrained set-based face recognition problem where each subject face is instantiated by a set of media (images and videos) instead of a single image. Naively aggregating information from all the media within a set would suffer from the large intra-set variance caused by heterogeneous factors (e.g., varying media modalities, poses and illuminations) and fail to learn discriminative face representations. A novel Multi-Prototype Network (MPNet) model is thus proposed to learn multiple prototype face representations adaptively from the media sets. Each learned prototype is representative for the subject face under certain condition in terms of pose, illumination and media modality. Instead of handcrafting the set partition for prototype learning, MPNet introduces a Dense SubGraph (DSG) learning sub-net that implicitly untangles inconsistent media and learns a number of representative prototypes. Qualitative and quantitative experiments clearly demonstrate superiority of the proposed model over state-of-the-arts.

[2] Video Re-ID文章

论文题目：Person Re-identification in Videos by Analyzing Spatio-Temporal Tubes

作者：Sk. Arif Ahmed, Debi Prosad Dogra, Heeseung Choi, Seungho Chae, Ig-Jae Kim

论文链接：https://arxiv.org/abs/1902.04856

摘要: Typical person re-identification frameworks search for k best matches in a gallery of images that are often collected in varying conditions. The gallery may contain image sequences when re-identification is done on videos. However, such a process is time consuming as re-identification has to be carried out multiple times. In this paper, we extract spatio-temporal sequences of frames (referred to as tubes) of moving persons and apply a multi-stage processing to match a given query tube with a gallery of stored tubes recorded through other cameras. Initially, we apply a binary classifier to remove noisy images from the input query tube. In the next step, we use a key-pose detection-based query minimization. This reduces the length of the query tube by removing redundant frames. Finally, a 3-stage hierarchical re-identification framework is used to rank the output tubes as per the matching scores. Experiments with publicly available video re-identification datasets reveal that our framework is better than state-of-the-art methods. It ranks the tubes with an increased CMC accuracy of 6-8% across multiple datasets. Also, our method significantly reduces the number of false positives. A new video re-identification dataset, named Tube-based Reidentification Video Dataset (TRiViD), has been prepared with an aim to help the re-identification research community

**2019-02-13**

[1] GluonCV 目标检测性能增加秘籍

论文题目：Bag of Freebies for Training Object Detection Neural Networks

作者：Zhi Zhang, Tong He, Hang Zhang, Zhongyuan Zhang, Junyuan Xie, Mu Li

论文链接：https://arxiv.org/abs/1902.04103

摘要: Comparing with enormous research achievements targeting better image classification models, efforts applied to object detector training are dwarfed in terms of popularity and universality. Due to significantly more complex network structures and optimization targets, various training strategies and pipelines are specifically designed for certain detection algorithms and no other. In this work, we explore universal tweaks that help boosting the performance of state-of-the-art object detection models to a new level without sacrificing inference speed. Our experiments indicate that these freebies can be as much as 5% absolute precision increase that everyone should consider applying to object detection training to a certain degree.

[2] Visual Grounding文章

论文题目：You Only Look & Listen Once: Towards Fast and Accurate Visual Grounding

作者：Chaorui Deng, Qi Wu, Guanghui Xu, Zhuliang Yu, Yanwu Xu, Kui Jia, Mingkui Tan

论文链接：https://arxiv.org/abs/1902.04213

代码链接：https://github.com/openblack/rvg

摘要: Visual Grounding (VG) aims to locate the most relevant region in an image, based on a flexible natural language query but not a pre-defined label, thus it can be a more useful technique than object detection in practice. Most state-of-the-art methods in VG operate in a two-stage manner, wherein the first stage an object detector is adopted to generate a set of object proposals from the input image and the second stage is simply formulated as a cross-modal matching problem that finds the best match between the language query and all region proposals. This is rather inefficient because there might be hundreds of proposals produced in the first stage that need to be compared in the second stage, not to mention this strategy performs inaccurately. In this paper, we propose an simple, intuitive and much more elegant one-stage detection based method that joints the region proposal and matching stage as a single detection network. The detection is conditioned on the input query with a stack of novel Relation-to-Attention modules that transform the image-to-query relationship to an relation map, which is used to predict the bounding box directly without proposing large numbers of useless region proposals. During the inference, our approach is about 20x ~ 30x faster than previous methods and, remarkably, it achieves 18% ~ 41% absolute performance improvement on top of the state-of-the-art results on several benchmark datasets.

[3] 在论文中画网络结构图的文章

论文题目：Net2Vis: Transforming Deep Convolutional Networks into Publication-Ready Visualizations

作者：Alex Bäuerle, Timo Ropinski

论文链接：https://arxiv.org/abs/1902.04394

代码链接：https://gitlab.com/Sparkier/Net2Vis

摘要: To properly convey neural network architectures in publications, appropriate visualization techniques are of great importance. While most current deep learning papers contain such visualizations, these are usually handcrafted, which results in a lack of a common visual grammar, as well as a significant time investment. Since these visualizations are often crafted just before publication, they are also prone to contain errors, might deviate from the actual architecture, and are sometimes ambiguous to interpret. Current automatic network visualization toolkits focus on debugging the network itself, and are therefore not ideal for generating publication-ready visualization, as they cater a different level of detail. Therefore, we present an approach to automate this process by translating network architectures specified in Python, into publication-ready network visualizations that can directly be embedded into any publication. To improve the readability of these visualizations, and in order to make them comparable, the generated visualizations obey to a visual grammar, which we have derived based on the analysis of existing network visualizations. Besides carefully crafted visual encodings, our grammar also incorporates abstraction through layer accumulation, as it is often done to reduce the complexity of the network architecture to be communicated. Thus, our approach not only reduces the time needed to generate publication-ready network visualizations, but also enables a unified and unambiguous visualization design.

[4] 3D Instance Segmentation文章

论文题目：MASC: Multi-scale Affinity with Sparse Convolution for 3D Instance Segmentation

作者：Chen Liu, Yasutaka Furukawa

论文链接：https://arxiv.org/abs/1902.04478

代码链接：https://github.com/art-programmer/MASC

摘要: We propose a new approach for 3D instance segmentation based on sparse convolution and point affinity prediction, which indicates the likelihood of two points belonging to the same instance. The proposed network, built upon submanifold sparse convolution [3], processes a voxelized point cloud and predicts semantic scores for each occupied voxel as well as the affinity between neighboring voxels at different scales. A simple yet effective clustering algorithm segments points into instances based on the predicted affinity and the mesh topology. The semantic for each instance is determined by the semantic prediction. Experiments show that our method outperforms the state-of-the-art instance segmentation methods by a large margin on the widely used ScanNet benchmark.

[5] 关于网络结构设计原则的文章

论文题目：Capacity allocation analysis of neural networks: A tool for principled architecture design

作者：Jonathan Donier

论文链接：https://arxiv.org/abs/1902.04485

摘要: Designing neural network architectures is a task that lies somewhere between science and art. For a given task, some architectures are eventually preferred over others, based on a mix of intuition, experience, experimentation and luck. For many tasks, the final word is attributed to the loss function, while for some others a further perceptual evaluation is necessary to assess and compare performance across models. In this paper, we introduce the concept of capacity allocation analysis, with the aim of shedding some light on what network architectures focus their modelling capacity on, when used on a given task. We focus more particularly on spatial capacity allocation, which analyzes a posteriori the effective number of parameters that a given model has allocated for modelling dependencies on a given point or region in the input space, in linear settings. We use this framework to perform a quantitative comparison between some classical architectures on various synthetic tasks. Finally, we consider how capacity allocation might translate in non-linear settings.


[6] AlphaZero开源实现ELF OpenGo

论文题目：ELF OpenGo: An Analysis and Open Reimplementation of AlphaZero

作者：Yuandong Tian, Jerry Ma, Qucheng Gong, Shubho Sengupta, Zhuoyuan Chen, James Pinkerton, C. Lawrence Zitnick

论文链接：https://arxiv.org/abs/1902.04522

代码链接：https://facebook.ai/developers/tools/elf-opengo

摘要: The AlphaGo, AlphaGo Zero, and AlphaZero series of algorithms are a remarkable demonstration of deep reinforcement learning's capabilities, achieving superhuman performance in the complex game of Go with progressively increasing autonomy. However, many obstacles remain in the understanding of and usability of these promising approaches by the research community. Toward elucidating unresolved mysteries and facilitating future research, we propose ELF OpenGo, an open-source reimplementation of the AlphaZero algorithm. ELF OpenGo is the first open-source Go AI to convincingly demonstrate superhuman performance with a perfect (20:0) record against global top professionals. We apply ELF OpenGo to conduct extensive ablation studies, and to identify and analyze numerous interesting phenomena in both the model training and in the gameplay inference procedures. Our code, models, selfplay datasets, and auxiliary data are publicly available.

[7] 实时语义分割文章

论文题目：Fast-SCNN: Fast Semantic Segmentation Network

作者：Rudra P K Poudel, Stephan Liwicki, Roberto Cipolla

论文链接：https://arxiv.org/abs/1902.04502

摘要: The encoder-decoder framework is state-of-the-art for offline semantic image segmentation. Since the rise in autonomous systems, real-time computation is increasingly desirable. In this paper, we introduce fast segmentation convolutional neural network (Fast-SCNN), an above real-time semantic segmentation model on high resolution image data (1024x2048px) suited to efficient computation on embedded devices with low memory. Building on existing two-branch methods for fast segmentation, we introduce our `learning to downsample' module which computes low-level features for multiple resolution branches simultaneously. Our network combines spatial detail at high resolution with deep features extracted at lower resolution, yielding an accuracy of 68.0% mean intersection over union at 123.5 frames per second on Cityscapes. We also show that large scale pre-training is unnecessary. We thoroughly validate our metric in experiments with ImageNet pre-training and the coarse labeled data of Cityscapes. Finally, we show even faster computation with competitive results on subsampled inputs, without any network modifications.

**2019-02-12**

[1] activity prediction文章

论文题目：Peeking into the Future: Predicting Future Person Activities and Locations in Videos

作者：Junwei Liang, Lu Jiang, Juan Carlos Niebles, Alexander Hauptmann, Li Fei-Fei

文章链接：https://arxiv.org/abs/1902.03748

摘要：Deciphering human behaviors to predict their future paths/trajectories and what they would do from videos is important in many applications. Motivated by this idea, this paper studies predicting a pedestrian's future path jointly with future activities. We propose an end-to-end, multi-task learning system utilizing rich visual features about the human behavioral information and interaction with their surroundings. To facilitate the training, the network is learned with two auxiliary tasks of predicting future activities and the location in which the activity will happen. Experimental results demonstrate our state-of-the-art performance over two public benchmarks on future trajectory prediction. Moreover, our method is able to produce meaningful future activity prediction in addition to the path. The result provides the first empirical evidence that a joint modeling of paths and activities benefits future path prediction.

[2] ICRA 2019 SLAM文章

论文题目：Visual SLAM: Why Bundle Adjust?

作者：Alvaro Parra Bustos, Tat-Jun Chin, Anders Eriksson, Ian Reid

文章链接：https://arxiv.org/abs/1902.03747

摘要：Bundle adjustment plays a vital role in feature-based monocular SLAM. In many modern SLAM pipelines, bundle adjustment is performed to estimate the 6DOF camera trajectory and 3D map (3D point cloud) from the input feature tracks. However, two fundamental weaknesses plague SLAM systems based on bundle adjustment. First, the need to carefully initialise bundle adjustment means that all variables, in particular the map, must be estimated as accurately as possible and maintained over time, which makes the overall algorithm cumbersome. Second, since estimating the 3D structure (which requires sufficient baseline) is inherent in bundle adjustment, the SLAM algorithm will encounter difficulties during periods of slow motion or pure rotational motion. 
We propose a different SLAM optimisation core: instead of bundle adjustment, we conduct rotation averaging to incrementally optimise only camera orientations. Given the orientations, we estimate the camera positions and 3D points via a quasi-convex formulation that can be solved efficiently and globally optimally. Our approach not only obviates the need to estimate and maintain the positions and 3D map at keyframe rate (which enables simpler SLAM systems), it is also more capable of handling slow motions or pure rotational motions.

[3] Dense Depth Estimation文章

论文题目：A Motion Free Approach to Dense Depth Estimation in Complex Dynamic Scene

作者：Suryansh Kumar, Ram Srivatsav Ghorakavi, Yuchao Dai, Hongdong Li

文章链接：https://arxiv.org/abs/1902.03791

摘要：Despite the recent success in per-frame monocular dense depth estimation of rigid scenes using deep learning methods, they fail to achieve similar success for complex dynamic scenes, such as MPI Sintel \cite{butler2012naturalistic}. Moreover, conventional geometric methods to address this problem using a piece-wise rigid scene model requires a reliable estimation of motion parameters for each local model, which is difficult to obtain and validate. In this work, we show that, given per-pixel optical flow correspondences between two consecutive frames and the sparse depth prior for the reference frame, we can recover the dense depth map for the successive frames without solving for motion parameters. By assigning the locally rigid structure to the piece-wise planar approximation of a dynamic scene which transforms as rigid as possible over frames, we demonstrate that we can bypass the motion estimation step. In essence, our formulation provides a new way to think and recover dense depth map of a complex dynamic scene which is recursive, incremental and motion free in nature and therefore, it can also be integrated with the modern neural network frameworks for large-scale depth-estimation applications. Our proposed method does not make any prior assumption about the rigidity of a dynamic scene, as a result, it is applicable to a wide range of scenarios. Experimental results show that our method can effectively provide the depth for the successive/multiple frames of a dynamic scene without using any motion parameters.

[4] SLAM文章

论文题目：UcoSLAM: Simultaneous Localization and Mapping by Fusion of KeyPoints and Squared Planar Markers

作者：Rafael Munoz-Salinas, Rafael Medina-Carnicer

文章链接：https://arxiv.org/abs/1902.03729

摘要：This paper proposes a novel approach for Simultaneous Localization and Mapping by fusing natural and artificial landmarks. Most of the SLAM approaches use natural landmarks (such as keypoints). However, they are unstable over time, repetitive in many cases or insufficient for a robust tracking (e.g. in indoor buildings). On the other hand, other approaches have employed artificial landmarks (such as squared fiducial markers) placed in the environment to help tracking and relocalization. We propose a method that integrates both approaches in order to achieve long-term robust tracking in many scenarios. 
Our method has been compared to the start-of-the-art methods ORB-SLAM2 and LDSO in the public dataset Kitti, Euroc-MAV, TUM and SPM, obtaining better precision, robustness and speed. Our tests also show that the combination of markers and keypoints achieves better accuracy than each one of them independently.

[5] 运动目标分割文章

论文题目：Towards Segmenting Everything That Moves

作者：Achal Dave, Pavel Tokmakov, Deva Ramanan

文章链接：https://arxiv.org/abs/1902.03715

摘要：Video analysis is the task of perceiving the world as it changes. Often, though, most of the world doesn't change all that much: it's boring. For many applications such as action detection or robotic interaction, segmenting all moving objects is a crucial first step. While this problem has been well-studied in the field of spatiotemporal segmentation, virtually none of the prior works use learning-based approaches, despite significant advances in single-frame instance segmentation. We propose the first deep-learning based approach for video instance segmentation. Our two-stream models' architecture is based on Mask R-CNN, but additionally takes optical flow as input to identify moving objects. It then combines the motion and appearance cues to correct motion estimation mistakes and capture the full extent of objects. We show state-of-the-art results on the Freiburg Berkeley Motion Segmentation dataset by a wide margin. One potential worry with learning-based methods is that they might overfit to the particular type of objects that they have been trained on. While current recognition systems tend to be limited to a "closed world" of N objects on which they are trained, our model seems to segment almost anything that moves.

[6] MOT+segmentation数据集

论文题目：MOTS: Multi-Object Tracking and Segmentation

作者：Paul Voigtlaender, Michael Krause, Aljosa Osep, Jonathon Luiten, Berin Balachandar Gnana Sekar, Andreas Geiger, Bastian Leibe

文章链接：https://arxiv.org/abs/1902.03604

摘要：This paper extends the popular task of multi-object tracking to multi-object tracking and segmentation (MOTS). Towards this goal, we create dense pixel-level annotations for two existing tracking datasets using a semi-automatic annotation procedure. Our new annotations comprise 70,430 pixel masks for 1,084 distinct objects (cars and pedestrians) in 10,870 video frames. For evaluation, we extend existing multi-object tracking metrics to this new task. Moreover, we propose a new baseline method which jointly addresses detection, tracking, and segmentation with a single convolutional network. We demonstrate the value of our datasets by achieving improvements in performance when training on MOTS annotations. We believe that our datasets, metrics and baseline will become a valuable resource towards developing multi-object tracking approaches that go beyond 2D bounding boxes.

[7] CVIU 2019 Face-SSD文章

论文题目：Registration-free Face-SSD: Single shot analysis of smiles, facial attributes, and affect in the wild

作者：Youngkyoon Jang, Hatice Gunes, Ioannis Patras

文章链接：https://arxiv.org/abs/1902.04042

摘要：In this paper, we present a novel single shot face-related task analysis method, called Face-SSD, for detecting faces and for performing various face-related (classification/regression) tasks including smile recognition, face attribute prediction and valence-arousal estimation in the wild. Face-SSD uses a Fully Convolutional Neural Network (FCNN) to detect multiple faces of different sizes and recognise/regress one or more face-related classes. Face-SSD has two parallel branches that share the same low-level filters, one branch dealing with face detection and the other one with face analysis tasks. The outputs of both branches are spatially aligned heatmaps that are produced in parallel - therefore Face-SSD does not require that face detection, facial region extraction, size normalisation, and facial region processing are performed in subsequent steps. Our contributions are threefold: 1) Face-SSD is the first network to perform face analysis without relying on pre-processing such as face detection and registration in advance - Face-SSD is a simple and a single FCNN architecture simultaneously performing face detection and face-related task analysis - those are conventionally treated as separate consecutive tasks; 2) Face-SSD is a generalised architecture that is applicable for various face analysis tasks without modifying the network structure - this is in contrast to designing task-specific architectures; and 3) Face-SSD achieves real-time performance (21 FPS) even when detecting multiple faces and recognising multiple classes in a given image. Experimental results show that Face-SSD achieves state-of-the-art performance in various face analysis tasks by reaching a recognition accuracy of 95.76% for smile detection, 90.29% for attribute prediction, and Root Mean Square (RMS) error of 0.44 and 0.39 for valence and arousal estimation.

[8] Biomedical Image Segmentation文章

论文题目：MultiResUNet : Rethinking the U-Net Architecture for Multimodal Biomedical Image Segmentation

作者：Nabil Ibtehaz, M. Sohel Rahman

文章链接：https://arxiv.org/abs/1902.04049

摘要：In recent years Deep Learning has brought about a breakthrough in Medical Image Segmentation. U-Net is the most prominent deep network in this regard, which has been the most popular architecture in the medical imaging community. Despite outstanding overall performance in segmenting multimodal medical images, from extensive experimentations on challenging datasets, we found out that the classical U-Net architecture seems to be lacking in certain aspects. Therefore, we propose some modifications to improve upon the already state-of-the-art U-Net model. Hence, following the modifications we develop a novel architecture MultiResUNet as the potential successor to the successful U-Net architecture. We have compared our proposed architecture MultiResUNet with the classical U-Net on a vast repertoire of multimodal medical images. Albeit slight improvements in the cases of ideal images, a remarkable gain in performance has been attained for challenging images. We have evaluated our model on five different datasets, each with their own unique challenges, and have obtained a relative improvement in performance of 10.15%, 5.07%, 2.63%, 1.41%, and 0.62% respectively.

[9] Face Recognition文章

论文题目：Cross-spectral Face Completion for NIR-VIS Heterogeneous Face Recognition

作者：Ran He, Jie Cao, Lingxiao Song, Zhenan Sun, Tieniu Tan

文章链接：https://arxiv.org/abs/1902.03565

摘要：Near infrared-visible (NIR-VIS) heterogeneous face recognition refers to the process of matching NIR to VIS face images. Current heterogeneous methods try to extend VIS face recognition methods to the NIR spectrum by synthesizing VIS images from NIR images. However, due to self-occlusion and sensing gap, NIR face images lose some visible lighting contents so that they are always incomplete compared to VIS face images. This paper models high resolution heterogeneous face synthesis as a complementary combination of two components, a texture inpainting component and pose correction component. The inpainting component synthesizes and inpaints VIS image textures from NIR image textures. The correction component maps any pose in NIR images to a frontal pose in VIS images, resulting in paired NIR and VIS textures. A warping procedure is developed to integrate the two components into an end-to-end deep network. A fine-grained discriminator and a wavelet-based discriminator are designed to supervise intra-class variance and visual quality respectively. One UV loss, two adversarial losses and one pixel loss are imposed to ensure synthesis results. We demonstrate that by attaching the correction component, we can simplify heterogeneous face synthesis from one-to-many unpaired image translation to one-to-one paired image translation, and minimize spectral and pose discrepancy during heterogeneous recognition. Extensive experimental results show that our network not only generates high-resolution VIS face images and but also facilitates the accuracy improvement of heterogeneous face recognition.

[10] 

论文题目：NeurAll: Towards a Unified Model for Visual Perception in Automated Driving

作者：Ganesh Sistu, Isabelle Leang, Sumanth Chennupati, Stefan Milz, Senthil Yogamani, Samir Rawashdeh

文章链接：https://arxiv.org/abs/1902.03589

摘要：Convolutional Neural Networks (CNNs) are successfully used for the important automotive visual perception tasks including object recognition, motion and depth estimation, visual SLAM, etc. However, these tasks are independently explored and modeled. In this paper, we propose a joint multi-task network design called NeurAll for learning all tasks simultaneously. Our main motivation is the computational efficiency achieved by sharing the expensive initial convolutional layers between all tasks. Indeed, the main bottleneck in automated driving systems is the limited processing power available on deployment hardware. There could be other benefits in improving accuracy for some tasks and it eases development effort. It also offers scalability to add more tasks leveraging existing features and achieving better generalization. We survey various CNN based solutions for visual perception tasks in automated driving. Then we propose a unified CNN model for the important tasks and discuss several advanced optimization and architecture design techniques to improve the baseline model. The paper is partly review and partly positional with demonstration of several preliminary results promising for future research. Firstly, we show that an efficient two-task model performing semantic segmentation and object detection achieves similar accuracies compared to separate models on various datasets with minimized runtime. We then illustrate that using depth regression as auxiliary task improves semantic segmentation and using multi-stream semantic segmentation outperforms one-stream semantic segmentation. The two-task network achieves 30 fps on an automotive grade low power SOC for 1280x384 image resolution

[11] 3D Hand Pose文章

论文题目：3D Hand Shape and Pose from Images in the Wild

作者：Adnane Boukhayma, Rodrigo de Bem, Philip H.S. Torr

文章链接：https://arxiv.org/abs/1902.03451

摘要：We present in this work the first end-to-end deep learning based method that predicts both 3D hand shape and pose from RGB images in the wild. Our network consists of the concatenation of a deep convolutional encoder, and a fixed model-based decoder. Given an input image, and optionally 2D joint detections obtained from an independent CNN, the encoder predicts a set of hand and view parameters. The decoder has two components: A pre-computed articulated mesh deformation hand model that generates a 3D mesh from the hand parameters, and a re-projection module controlled by the view parameters that projects the generated hand into the image domain. We show that using the shape and pose prior knowledge encoded in the hand model within a deep learning framework yields state-of-the-art performance in 3D pose prediction from images on standard benchmarks, and produces geometrically valid and plausible 3D reconstructions. Additionally, we show that training with weak supervision in the form of 2D joint annotations on datasets of images in the wild, in conjunction with full supervision in the form of 3D joint annotations on limited available datasets allows for good generalization to 3D shape and pose predictions on images in the wild.

[12] 网络结构压缩文章

论文题目：Architecture Compression

作者：Anubhav Ashok

文章链接：https://arxiv.org/abs/1902.03326

摘要：In this paper we propose a novel approach to model compression termed Architecture Compression. Instead of operating on the weight or filter space of the network like classical model compression methods, our approach operates on the architecture space. A 1-D CNN encoder-decoder is trained to learn a mapping from discrete architecture space to a continuous embedding and back. Additionally, this embedding is jointly trained to regress accuracy and parameter count in order to incorporate information about the architecture's effectiveness on the dataset. During the compression phase, we first encode the network and then perform gradient descent in continuous space to optimize a compression objective function that maximizes accuracy and minimizes parameter count. The final continuous feature is then mapped to a discrete architecture using the decoder. We demonstrate the merits of this approach on visual recognition tasks such as CIFAR-10, CIFAR-100, Fashion-MNIST and SVHN and achieve a greater than 20x compression on CIFAR-10.

**2019-02-11**

[1] Action Prediction文章

论文题目：Skeleton-Based Online Action Prediction Using Scale Selection Network

作者：Jun Liu, Amir Shahroudy, Gang Wang, Ling-Yu Duan, Alex C. Kot

文章链接：https://arxiv.org/abs/1902.03084

摘要：Action prediction is to recognize the class label of an ongoing activity when only a part of it is observed. In this paper, we focus on online action prediction in streaming 3D skeleton sequences. A dilated convolutional network is introduced to model the motion dynamics in temporal dimension via a sliding window over the temporal axis. Since there are significant temporal scale variations in the observed part of the ongoing action at different time steps, a novel window scale selection method is proposed to make our network focus on the performed part of the ongoing action and try to suppress the possible incoming interference from the previous actions at each step. An activation sharing scheme is also proposed to handle the overlapping computations among the adjacent time steps, which enables our framework to run more efficiently. Moreover, to enhance the performance of our framework for action prediction with the skeletal input data, a hierarchy of dilated tree convolutions are also designed to learn the multi-level structured semantic representations over the skeleton joints at each frame. Our proposed approach is evaluated on four challenging datasets. The extensive experiments demonstrate the effectiveness of our method for skeleton-based online action prediction.

[2] 3D Human Pose Estimation文章

论文题目：3D Human Pose Estimation from Deep Multi-View 2D Pose

作者：Steven Schwarcz, Thomas Pollard

文章链接：https://arxiv.org/abs/1902.02841

摘要：Human pose estimation - the process of recognizing a human's limb positions and orientations in a video - has many important applications including surveillance, diagnosis of movement disorders, and computer animation. While deep learning has lead to great advances in 2D and 3D pose estimation from single video sources, the problem of estimating 3D human pose from multiple video sensors with overlapping fields of view has received less attention. When the application allows use of multiple cameras, 3D human pose estimates may be greatly improved through fusion of multi-view pose estimates and observation of limbs that are fully or partially occluded in some views. Past approaches to multi-view 3D pose estimation have used probabilistic graphical models to reason over constraints, including per-image pose estimates, temporal smoothness, and limb length. In this paper, we present a pipeline for multi-view 3D pose estimation of multiple individuals which combines a state-of-art 2D pose detector with a factor graph of 3D limb constraints optimized with belief propagation. We evaluate our results on the TUM-Campus and Shelf datasets for multi-person 3D pose estimation and show that our system significantly out-performs the previous state-of-the-art with a simpler model of limb dependency.

[3] 单目标跟踪文章

论文题目：SiamVGG: Visual Tracking using Deeper Siamese Networks

作者：Yuhong Li, Xiaofan Zhang

文章链接：https://arxiv.org/abs/1902.02804

摘要：Recently, we have seen a rapid development of Deep Neural Network (DNN) based visual tracking solutions. Some trackers combine the DNN-based solutions with Discriminative Correlation Filters (DCF) to extract semantic features and successfully deliver the state-of-the-art tracking accuracy. However, these solutions are highly compute-intensive, which require long processing time, resulting unsecured real-time performance. To deliver both high accuracy and reliable real-time performance, we propose a novel tracker called SiamVGG. It combines a Convolutional Neural Network (CNN) backbone and a cross-correlation operator, and takes advantage of the features from exemplary images for more accurate object tracking. 
The architecture of SiamVGG is customized from VGG-16, with the parameters shared by both exemplary images and desired input video frames. 
We demonstrate the proposed SiamVGG on OTB-2013/50/100 and VOT 2015/2016/2017 datasets with the state-of-the-art accuracy while maintaining a decent real-time performance of 50 FPS running on a GTX 1080Ti. Our design can achieve 2% higher Expected Average Overlap (EAO) compared to the ECO and C-COT in VOT2017 Challenge.













