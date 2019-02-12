# CV-arXiv-Daily

**分享计算机视觉每天的arXiv文章，主要集中在目标检测，单目标跟踪，多目标跟踪，人体行为识别，人体姿态估计与跟踪，行人重识别，模型搜索等。每周周末会将本周的Archive起来**

[2019-01-23~2019-01-26](2019/2019.01.23-2019.01.26.md)

[2019-01-28~2019-02-01](2019/2019.01.28-2019.02.01.md)

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













