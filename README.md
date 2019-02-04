# CV-arXiv-Daily

**分享计算机视觉每天的arXiv文章，主要集中在目标检测，单目标跟踪，多目标跟踪，人体行为识别，人体姿态估计与跟踪，行人重识别，模型搜索等。每周周末会将本周的Archive起来**

[2019-01-23~2019-01-26](2019/2019.01.23-2019.01.26.md)

**2019-02-04**

[1] ACM Multimedia 2018 图像检索oral文章

论文题目：Deep Triplet Quantization

论文链接：https://arxiv.org/abs/1902.00153

摘要: Deep hashing establishes efficient and effective image retrieval by end-to-end learning of deep representations and hash codes from similarity data. We present a compact coding solution, focusing on deep learning to quantization approach that has shown superior performance over hashing solutions for similarity retrieval. We propose Deep Triplet Quantization (DTQ), a novel approach to learning deep quantization models from the similarity triplets. To enable more effective triplet training, we design a new triplet selection approach, Group Hard, that randomly selects hard triplets in each image group. To generate compact binary codes, we further apply a triplet quantization with weak orthogonality during triplet training. The quantization loss reduces the codebook redundancy and enhances the quantizability of deep representations through back-propagation. Extensive experiments demonstrate that DTQ can generate high-quality and compact binary codes, which yields state-of-the-art image retrieval performance on three benchmark datasets, NUS-WIDE, CIFAR-10, and MS-COCO.

[2] ICLR compressed NAS文章

论文题目：Learnable Embedding Space for Efficient Neural Architecture Compression

论文链接：https://arxiv.org/abs/1902.00383

摘要: We propose a method to incrementally learn an embedding space over the domain of network architectures, to enable the careful selection of architectures for evaluation during compressed architecture search. Given a teacher network, we search for a compressed network architecture by using Bayesian Optimization (BO) with a kernel function defined over our proposed embedding space to select architectures for evaluation. We demonstrate that our search algorithm can significantly outperform various baseline methods, such as random search and reinforcement learning (Ashok et al., 2018). The compressed architectures found by our method are also better than the state-of-the-art manually-designed compact architecture ShuffleNet (Zhang et al., 2018). We also demonstrate that the learned embedding space can be transferred to new settings for architecture search, such as a larger teacher network or a teacher network in a different architecture family, without any training.

[3] ciFAIR数据集（无重复样本的CIFAR）

论文题目：Do we train on test data? Purging CIFAR of near-duplicates

论文链接：https://arxiv.org/abs/1902.00423

代码链接：https://cvjena.github.io/cifair/

摘要: We find that 3.3% and 10% of the images from the CIFAR-10 and CIFAR-100 test sets, respectively, have duplicates in the training set. This may incur a bias on the comparison of image recognition techniques with respect to their generalization capability on these heavily benchmarked datasets. To eliminate this bias, we provide the "fair CIFAR" (ciFAIR) dataset, where we replaced all duplicates in the test sets with new images sampled from the same domain. The training set remains unchanged, in order not to invalidate pre-trained models. We then re-evaluate the classification performance of various popular state-of-the-art CNN architectures on these new test sets to investigate whether recent research has overfitted to memorizing data instead of learning abstract concepts. Fortunately, this does not seem to be the case yet. The ciFAIR dataset and pre-trained models are available at this https URL, where we also maintain a leaderboard.

[4] 行人轨迹数据集

论文题目：Top-view Trajectories: A Pedestrian Dataset of Vehicle-Crowd Interaction from Controlled Experiments and Crowded Campus

论文链接：https://arxiv.org/abs/1902.00487

摘要: Predicting the collective motion of a group of pedestrians (a crowd) under the vehicle influence is essential for the development of autonomous vehicles to deal with mixed urban scenarios where interpersonal interaction and vehicle-crowd interaction (VCI) are significant. This usually requires a model that can describe individual pedestrian motion under the influence of nearby pedestrians and the vehicle. This study proposed two pedestrian trajectory dataset, CITR dataset and DUT dataset, so that the pedestrian motion models can be further calibrated and verified, especially when vehicle influence on pedestrians plays an important role. CITR dataset consists of experimentally designed fundamental VCI scenarios (front, back, and lateral VCIs) and provides unique ID for each pedestrian, which is suitable for exploring a specific aspect of VCI. DUT dataset gives two ordinary and natural VCI scenarios in crowded university campus, which can be used for more general purpose VCI exploration. The trajectories of pedestrians, as well as vehicles, were extracted by processing video frames that come from a down-facing camera mounted on a hovering drone as the recording equipment. The final trajectories were refined by a Kalman Filter, in which the pedestrian velocity was also estimated. The statistics of the velocity magnitude distribution demonstrated the validity of the proposed dataset. In total, there are approximate 340 pedestrian trajectories in CITR dataset and 1793 pedestrian trajectories in DUT dataset. The dataset is available at GitHub.



**2019-02-01**

[1] 一种新颖的3D Reconstruction方法

论文题目：Pix2Vox: Context-aware 3D Reconstruction from Single and Multi-view Images

论文链接：https://arxiv.org/abs/1901.11153

摘要: Recovering the 3D representation of an object from single-view or multi-view RGB images by deep neural networks has attracted increasing attention in the past few years. Several mainstream works (e.g., 3D-R2N2) use recurrent neural networks (RNNs) to fuse multiple feature maps extracted from input images sequentially. However, when given the same set of input images with different orders, RNN-based approaches are unable to produce consistent reconstruction results. Moreover, due to long-term memory loss, RNNs cannot fully exploit input images to refine reconstruction results. To solve these problems, we propose a novel framework for single-view and multi-view 3D reconstruction, named Pix2Vox. By using a well-designed encoder-decoder, it generates a coarse 3D volume from each input image. Then, a context-aware fusion module is introduced to adaptively select high-quality reconstructions for each part (e.g., table legs) from different coarse 3D volumes to obtain a fused 3D volume. Finally, a refiner further refines the fused 3D volume to generate the final output. Experimental results on the ShapeNet and Pascal 3D+ benchmarks indicate that the proposed Pix2Vox outperforms state-of-the-arts by a large margin. Furthermore, the proposed method is 24 times faster than 3D-R2N2 in terms of backward inference time. The experiments on ShapeNet unseen 3D categories have shown the superior generalization abilities of our method.

[2] Superpixel Segmentation文章

论文题目：Texture-Aware Superpixel Segmentation

论文链接：https://arxiv.org/abs/1901.11111

摘要: Most superpixel methods are based on spatial and color measures at the pixel level. Therefore, they can highly fail to group pixels with similar local texture properties, and need fine parameter tuning to balance the two measures. In this paper, we address these issues with a new Texture-Aware SuperPixel (TASP) segmentation method. TASP locally adjusts its spatial regularity constraint according to the feature variance to accurately segment both smooth and textured areas. A new pixel to superpixel patch-based distance is also proposed to ensure texture homogeneity within created regions. TASP substantially outperforms the segmentation accuracy of state-of-the-art methods on both natural color and texture images.

[3] 

论文题目：Capturing Object Detection Uncertainty in Multi-Layer Grid Maps

论文链接：https://arxiv.org/abs/1901.11284

摘要: We propose a deep convolutional object detector for automated driving applications that also estimates classification, pose and shape uncertainty of each detected object. The input consists of a multi-layer grid map which is well-suited for sensor fusion, free-space estimation and machine learning. Based on the estimated pose and shape uncertainty we approximate object hulls with bounded collision probability which we find helpful for subsequent trajectory planning tasks. We train our models based on the KITTI object detection data set. In a quantitative and qualitative evaluation some models show a similar performance and superior robustness compared to previously developed object detectors. However, our evaluation also points to undesired data set properties which should be addressed when training data-driven models or creating new data sets.

[4] 网络压缩文章

论文题目：Partition Pruning: Parallelization-Aware Pruning for Deep Neural Networks

论文链接：https://arxiv.org/abs/1901.11391

摘要: Parameters of recent neural networks require a huge amount of memory. These parameters are used by neural networks to perform machine learning tasks when processing inputs. To speed up inference, we develop Partition Pruning, an innovative scheme to reduce the parameters used while taking into consideration parallelization. We evaluated the performance and energy consumption of parallel inference of partitioned models, which showed a 7.72x speed up of performance and a 2.73x reduction in the energy used for computing pruned layers of TinyVGG16 in comparison to running the unpruned model on a single accelerator. In addition, our method showed a limited reduction some numbers in accuracy while partitioning fully connected layers.

[5] Hotel Recognition数据集

论文题目：Hotels-50K: A Global Hotel Recognition Dataset

论文链接：https://arxiv.org/abs/1901.11397

摘要: Recognizing a hotel from an image of a hotel room is important for human trafficking investigations. Images directly link victims to places and can help verify where victims have been trafficked, and where their traffickers might move them or others in the future. Recognizing the hotel from images is challenging because of low image quality, uncommon camera perspectives, large occlusions (often the victim), and the similarity of objects (e.g., furniture, art, bedding) across different hotel rooms. 
To support efforts towards this hotel recognition task, we have curated a dataset of over 1 million annotated hotel room images from 50,000 hotels. These images include professionally captured photographs from travel websites and crowd-sourced images from a mobile application, which are more similar to the types of images analyzed in real-world investigations. We present a baseline approach based on a standard network architecture and a collection of data-augmentation approaches tuned to this problem domain.

**2019-01-31**

[1] ECCV2018 PIRM-SR Challenge 超分辨率比赛亚军方案

论文题目：Multi-Scale Recursive and Perception-Distortion Controllable Image Super-Resolution

论文链接：https://arxiv.org/abs/1809.10711

代码链接：https://github.com/pnavarre/pirm-sr-2018

摘要: We describe our solution for the PIRM Super-Resolution Challenge 2018 where we achieved the 2nd best perceptual quality for average RMSE<=16, 5th best for RMSE<=12.5, and 7th best for RMSE<=11.5. We modify a recently proposed Multi-Grid Back-Projection (MGBP) architecture to work as a generative system with an input parameter that can control the amount of artificial details in the output. We propose a discriminator for adversarial training with the following novel properties: it is multi-scale that resembles a progressive-GAN; it is recursive that balances the architecture of the generator; and it includes a new layer to capture significant statistics of natural images. Finally, we propose a training strategy that avoids conflicts between reconstruction and perceptual losses. Our configuration uses only 281k parameters and upscales each image of the competition in 0.2s in average.

[2] Re-ID中的对抗攻击

论文题目：Adversarial Metric Attack for Person Re-identification

论文链接：https://arxiv.org/abs/1901.10650

摘要: Person re-identification (re-ID) has attracted much attention recently due to its great importance in video surveillance. In general, distance metrics used to identify two person images are expected to be robust under various appearance changes. However, our work observes the extreme vulnerability of existing distance metrics to adversarial examples, generated by simply adding human-imperceptible perturbations to person images. Hence, the security danger is dramatically increased when deploying commercial re-ID systems in video surveillance, especially considering the highly strict requirement of public safety.
Although adversarial examples have been extensively applied for classification analysis, it is rarely studied in metric analysis like person re-identification. The most likely reason is the natural gap between the training and testing of re-ID networks, that is, the predictions of a re-ID network cannot be directly used during testing without an effective metric. In this work, we bridge the gap by proposing Adversarial Metric Attack, a parallel methodology to adversarial classification attacks, which can effectively generate adversarial examples for re-ID. Comprehensive experiments clearly reveal the adversarial effects in re-ID systems. Moreover, by benchmarking various adversarial settings, we expect that our work can facilitate the development of robust feature learning with the experimental conclusions we have drawn.

[3] 

论文题目：Blurred Images Lead to Bad Local Minima

论文链接：https://arxiv.org/abs/1901.10788

摘要: Blurred Images Lead to Bad Local Minima

[4] Active Learning for LiDAR 3D Object Detector

论文题目：Deep Active Learning for Efficient Training of a LiDAR 3D Object Detector

论文链接：https://arxiv.org/abs/1901.10609

摘要: Training a deep object detector for autonomous driving requires a huge amount of labeled data. While recording data via on-board sensors such as camera or LiDAR is relatively easy, annotating data is very tedious and time-consuming, especially when dealing with 3D LiDAR points or radar data. Active learning has the potential to minimize human annotation efforts while maximizing the object detector's performance. In this work, we propose an active learning method to train a LiDAR 3D object detector with the least amount of labeled training data necessary. The detector leverages 2D region proposals generated from the RGB images to reduce the search space of objects and speed up the learning process. Experiments show that our proposed method works under different uncertainty estimations and query functions, and can save up to 60% of the labeling efforts while reaching the same network performance.

[5] 3D pose文章

论文题目：View Invariant 3D Human Pose Estimation

论文链接：https://arxiv.org/abs/1901.10841

摘要: The recent success of deep networks has significantly advanced 3D human pose estimation from 2D images. The diversity of capturing viewpoints and the flexibility of the human poses, however, remain some significant challenges. In this paper, we propose a view invariant 3D human pose estimation module to alleviate the effects of viewpoint diversity. The framework consists of a base network, which provides an initial estimation of a 3D pose, a view-invariant hierarchical correction network (VI-HC) on top of that to learn the 3D pose refinement under consistent views, and a view-invariant discriminative network (VID) to enforce high-level constraints over body configurations. In VI-HC, the initial 3D pose inputs are automatically transformed to consistent views for further refinements at the global body and local body parts level, respectively. For the VID, under consistent viewpoints, we use adversarial learning to differentiate between estimated poses and real poses to avoid implausible 3D poses. Experimental results demonstrate that the consistent viewpoints can dramatically enhance the performance. Our module shows robustness for different 3D pose base networks and achieves a significant improvement (about 9%) over a powerful baseline on the public 3D pose estimation benchmark Human3.6M.

[6] 

论文题目：Pixelated Semantic Colorization

论文链接：https://arxiv.org/abs/1901.10889

摘要: While many image colorization algorithms have recently shown the capability of producing plausible color versions from gray-scale photographs, they still suffer from limited semantic understanding. To address this shortcoming, we propose to exploit pixelated object semantics to guide image colorization. The rationale is that human beings perceive and distinguish colors based on the semantic categories of objects. Starting from an autoregressive model, we generate image color distributions, from which diverse colored results are sampled. We propose two ways to incorporate object semantics into the colorization model: through a pixelated semantic embedding and a pixelated semantic generator. Specifically, the proposed convolutional neural network includes two branches. One branch learns what the object is, while the other branch learns the object colors. The network jointly optimizes a color embedding loss, a semantic segmentation loss and a color generation loss, in an end-to-end fashion. Experiments on PASCAL VOC2012 and COCO-stuff reveal that our network, when trained with semantic segmentation labels, produces more realistic and finer results compared to the colorization state-of-the-art.

[7] 

论文题目：Generative Adversarial Network with Multi-Branch Discriminator for Cross-Species Image-to-Image Translation

论文链接：https://arxiv.org/abs/1901.10895

摘要: Current approaches have made great progress on image-to-image translation tasks benefiting from the success of image synthesis methods especially generative adversarial networks (GANs). However, existing methods are limited to handling translation tasks between two species while keeping the content matching on the semantic level. A more challenging task would be the translation among more than two species. To explore this new area, we propose a simple yet effective structure of a multi-branch discriminator for enhancing an arbitrary generative adversarial architecture (GAN), named GAN-MBD. It takes advantage of the boosting strategy to break a common discriminator into several smaller ones with fewer parameters, which can enhance the generation and synthesis abilities of GANs efficiently and effectively. Comprehensive experiments show that the proposed multi-branch discriminator can dramatically improve the performance of popular GANs on cross-species image-to-image translation tasks while reducing the number of parameters for computation. The code and some datasets are attached as supplementary materials for reference.

[8] AAAI-19 Spotlight 超分辨率文章

论文题目：Multigrid Backprojection Super-Resolution and Deep Filter Visualization

论文链接：https://arxiv.org/abs/1809.09326

摘要: We introduce a novel deep-learning architecture for image upscaling by large factors (e.g. 4x, 8x) based on examples of pristine high-resolution images. Our target is to reconstruct high-resolution images from their downscale versions. The proposed system performs a multi-level progressive upscaling, starting from small factors (2x) and updating for higher factors (4x and 8x). The system is recursive as it repeats the same procedure at each level. It is also residual since we use the network to update the outputs of a classic upscaler. The network residuals are improved by Iterative Back-Projections (IBP) computed in the features of a convolutional network. To work in multiple levels we extend the standard back-projection algorithm using a recursion analogous to Multi-Grid algorithms commonly used as solvers of large systems of linear equations. We finally show how the network can be interpreted as a standard upsampling-and-filter upscaler with a space-variant filter that adapts to the geometry. This approach allows us to visualize how the network learns to upscale. Finally, our system reaches state of the art quality for models with relatively few number of parameters.


**2019-01-30**

[1] 香侬科技横扫 13 项中文自然语言任务记录的Glyce

论文题目：Glyce: Glyph-vectors for Chinese Character Representations

论文链接：https://arxiv.org/abs/1901.10125

摘要: It is intuitive that NLP tasks for logographic languages like Chinese should benefit from the use of the glyph information in those languages. However, due to the lack of rich pictographic evidence in glyphs and the weak generalization ability of standard computer vision models on character data, an effective way to utilize the glyph information remains to be found. 
In this paper, we address this gap by presenting the Glyce, the glyph-vectors for Chinese character representations. We make three major innovations: (1) We use historical Chinese scripts (e.g., bronzeware script, seal script, traditional Chinese, etc) to enrich the pictographic evidence in characters; (2) We design CNN structures tailored to Chinese character image processing; and (3) We use image-classification as an auxiliary task in a multi-task learning setup to increase the model's ability to generalize. 
For the first time, we show that glyph-based models are able to consistently outperform word/char ID-based models in a wide range of Chinese NLP tasks. Using Glyce, we are able to achieve the state-of-the-art performances on 13 (almost all) Chinese NLP tasks, including (1) character-Level language modeling, (2) word-Level language modeling, (3) Chinese word segmentation, (4) name entity recognition, (5) part-of-speech tagging, (6) dependency parsing, (7) semantic role labeling, (8) sentence semantic similarity, (9) sentence intention identification, (10) Chinese-English machine translation, (11) sentiment analysis, (12) document classification and (13) discourse parsing

[2] 

论文题目：Diversity in Faces

论文链接：https://arxiv.org/abs/1901.10436

摘要:  Face recognition is a long standing challenge in the field of Artificial Intelligence (AI). The goal is to create systems that accurately detect, recognize, verify, and understand human faces. There are significant technical hurdles in making these systems accurate, particularly in unconstrained settings due to confounding factors related to pose, resolution, illumination, occlusion, and viewpoint. However, with recent advances in neural networks, face recognition has achieved unprecedented accuracy, largely built on data-driven deep learning methods. While this is encouraging, a critical aspect that is limiting facial recognition accuracy and fairness is inherent facial diversity. Every face is different. Every face reflects something unique about us. Aspects of our heritage - including race, ethnicity, culture, geography - and our individual identify - age, gender, and other visible manifestations of self-expression, are reflected in our faces. We expect face recognition to work equally accurately for every face. Face recognition needs to be fair. As we rely on data-driven methods to create face recognition technology, we need to ensure necessary balance and coverage in training data. However, there are still scientific questions about how to represent and extract pertinent facial features and quantitatively measure facial diversity. Towards this goal, Diversity in Faces (DiF) provides a data set of one million annotated human face images for advancing the study of facial diversity. The annotations are generated using ten well-established facial coding schemes from the scientific literature. The facial coding schemes provide human-interpretable quantitative measures of facial features. We believe that by making the extracted coding schemes available on a large set of faces, we can accelerate research and development towards creating more fair and accurate facial recognition systems. 

[3] PA-GAN

论文题目：PA-GAN: Improving GAN Training by Progressive Augmentation

论文链接：https://arxiv.org/abs/1901.10422

摘要:  Training of Generative Adversarial Networks (GANs) is notoriously fragile, which partially attributed to the discriminator performing well very quickly; its loss converges to zero, providing no reliable backpropagation signal to the generator. In this work we introduce a new technique - progressive augmentation of GANs (PA-GAN) - that helps to mitigate this issue and thus improve the GAN training. The key idea is to gradually increase the task difficulty of the discriminator by progressively augmenting its input or feature space, enabling continuous learning of the generator. We show that the proposed progressive augmentation preserves the original GAN objective, does not bias the optimality of the discriminator and encourages the healthy competition between the generator and discriminator, leading to a better-performing generator. We experimentally demonstrate the effectiveness of PA-GAN across different architectures and on multiple benchmarks for the image generation task. 

[4] TPAMI2019 无监督Re-ID新文

论文题目：Unsupervised Person Re-identification by Deep Asymmetric Metric Embedding

论文链接：https://arxiv.org/abs/1901.10177

摘要: Person re-identification (Re-ID) aims to match identities across non-overlapping camera views. Researchers have proposed many supervised Re-ID models which require quantities of cross-view pairwise labelled data. This limits their scalabilities to many applications where a large amount of data from multiple disjoint camera views is available but unlabelled. Although some unsupervised Re-ID models have been proposed to address the scalability problem, they often suffer from the view-specific bias problem which is caused by dramatic variances across different camera views, e.g., different illumination, viewpoints and occlusion. The dramatic variances induce specific feature distortions in different camera views, which can be very disturbing in finding cross-view discriminative information for Re-ID in the unsupervised scenarios, since no label information is available to help alleviate the bias. We propose to explicitly address this problem by learning an unsupervised asymmetric distance metric based on cross-view clustering. The asymmetric distance metric allows specific feature transformations for each camera view to tackle the specific feature distortions. We then design a novel unsupervised loss function to embed the asymmetric metric into a deep neural network, and therefore develop a novel unsupervised deep framework named the DEep Clustering-based Asymmetric MEtric Learning (DECAMEL). In such a way, DECAMEL jointly learns the feature representation and the unsupervised asymmetric metric. DECAMEL learns a compact cross-view cluster structure of Re-ID data, and thus help alleviate the view-specific bias and facilitate mining the potential cross-view discriminative information for unsupervised Re-ID. Extensive experiments on seven benchmark datasets whose sizes span several orders show the effectiveness of our framework. 

[5] 双流多任务Fashion Recognition

论文题目：Two-Stream Multi-Task Network for Fashion Recognition

论文链接：https://arxiv.org/abs/1901.10172
 
摘要:  In this paper, we present a two-stream multi-task network for fashion recognition. This task is challenging as fashion clothing always contain multiple attributes, which need to be predicted simultaneously for real-time industrial systems. To handle these challenges, we formulate fashion recognition into a multi-task learning problem, including landmark detection, category and attribute classifications, and solve it with the proposed deep convolutional neural network. We design two knowledge sharing strategies which enable information transfer between tasks and improve the overall performance. The proposed model achieves state-of-the-art results on large-scale fashion dataset comparing to the existing methods, which demonstrates its great effectiveness and superiority for fashion recognition. 

[6] 基于注意力机制的单目深度估计

论文题目：Attention-based Context Aggregation Network for Monocular Depth Estimation

论文链接：https://arxiv.org/abs/1901.10137

摘要: Depth estimation is a traditional computer vision task, which plays a crucial role in understanding 3D scene geometry. Recently, deep-convolutional-neural-networks based methods have achieved promising results in the monocular depth estimation field. Specifically, the framework that combines the multi-scale features extracted by the dilated convolution based block (atrous spatial pyramid pooling, ASPP) has gained the significant improvement in the dense labeling task. However, the discretized and predefined dilation rates cannot capture the continuous context information that differs in diverse scenes and easily introduce the grid artifacts in depth estimation. In this paper, we propose an attention-based context aggregation network (ACAN) to tackle these difficulties. Based on the self-attention model, ACAN adaptively learns the task-specific similarities between pixels to model the context information. First, we recast the monocular depth estimation as a dense labeling multi-class classification problem. Then we propose a soft ordinal inference to transform the predicted probabilities to continuous depth values, which can reduce the discretization error (about 1% decrease in RMSE). Second, the proposed ACAN aggregates both the image-level and pixel-level context information for depth estimation, where the former expresses the statistical characteristic of the whole image and the latter extracts the long-range spatial dependencies for each pixel. Third, for further reducing the inconsistency between the RGB image and depth map, we construct an attention loss to minimize their information entropy. We evaluate on public monocular depth-estimation benchmark datasets (including NYU Depth V2, KITTI). The experiments demonstrate the superiority of our proposed ACAN and achieve the competitive results with the state of the arts. 

[7] Re-ID文章，附代码

论文题目：Discovering Underlying Person Structure Pattern with Relative Local Distance for Person Re-identification

论文链接：https://arxiv.org/abs/1901.10100

代码地址：https://github.com/Wanggcong/RLD_codes

摘要: Modeling the underlying person structure for person re-identification (re-ID) is difficult due to diverse deformable poses, changeable camera views and imperfect person detectors. How to exploit underlying person structure information without extra annotations to improve the performance of person re-ID remains largely unexplored. To address this problem, we propose a novel Relative Local Distance (RLD) method that integrates a relative local distance constraint into convolutional neural networks (CNNs) in an end-to-end way. It is the first time that the relative local constraint is proposed to guide the global feature representation learning. Specially, a relative local distance matrix is computed by using feature maps and then regarded as a regularizer to guide CNNs to learn a structure-aware feature representation. With the discovered underlying person structure, the RLD method builds a bridge between the global and local feature representation and thus improves the capacity of feature representation for person re-ID. Furthermore, RLD also significantly accelerates deep network training compared with conventional methods. The experimental results show the effectiveness of RLD on the CUHK03, Market-1501, and DukeMTMC-reID datasets. 




**2019-01-29**

[1] 图森发布的SimpleDet框架

链接：https://github.com/TuSimple/simpledet

摘要: 
FP16 training for memory saving and up to 2.5X acceleration
Highly scalable distributed training available out of box
Full coverage of state-of-the-art models including FasterRCNN, MaskRCNN, CascadeRCNN, RetinaNet and TridentNet
Extensive feature set including large batch BN, deformable convolution, soft NMS, multi-scale train/test
Modular design for coding-free exploration of new experiment settings

[2] 6D Object Pose Estimation文章

论文题目：6D Object Pose Estimation Based on 2D Bounding Box

论文链接：https://arxiv.org/abs/1901.09366

摘要: In this paper, we present a simple but powerful method to tackle the problem of estimating the 6D pose of objects from a single RGB image. Our system trains a novel convolutional neural network to regress the unit quaternion, which represents the 3D rotation, from the partial image inside the bounding box returned by 2D detection systems. Then we propose an algorithm we call Bounding Box Equation to efficiently and accurately obtain the 3D translation, using 3D rotation and 2D bounding box. Considering that the quadratic sum of the quaternion's four elements equals to one, we add a normalization layer to keep the network's output on the unit sphere and put forward a special loss function for unit quaternion regression. We evaluate our method on the LineMod dataset and experiment shows that our approach outperforms base-line and some state of the art methods.

[3] 单目深度估计综述

论文题目：Monocular Depth Estimation: A Survey

论文链接：https://arxiv.org/abs/1901.09402

摘要: Monocular depth estimation is often described as an ill-posed and inherently ambiguous problem. Estimating depth from 2D images is a crucial step in scene reconstruction, 3Dobject recognition, segmentation, and detection. The problem can be framed as: given a single RGB image as input, predict a dense depth map for each pixel. This problem is worsened by the fact that most scenes have large texture and structural variations, object occlusions, and rich geometric detailing. All these factors contribute to difficulty in accurate depth estimation. In this paper, we review five papers that attempt to solve the depth estimation problem with various techniques including supervised, weakly-supervised, and unsupervised learning techniques. We then compare these papers and understand the improvements made over one another. Finally, we explore potential improvements that can aid to better solve this problem.

[4] 时空行为识别综述

论文题目：Spatio-temporal Action Recognition: A Survey

论文链接：https://arxiv.org/abs/1901.09403

摘要: The task of action recognition or action detection involves analyzing videos and determining what action or motion is being performed. The primary subject of these videos are predominantly humans performing some action. However, this requirement can be relaxed to generalize over other subjects such as animals or robots. The applications can range from anywhere between human-computer inter-action to automated video editing proposals. When we consider spatiotemporal action recognition, we deal with action localization. This task not only involves determining what action is being performed but also when and where itis being performed in said video. This paper aims to survey the plethora of approaches and algorithms attempted to solve this task, give a comprehensive comparison between them, explore various datasets available for the problem, and determine the most promising approaches.

[5] 人脸识别性能评测工具包

论文题目：Open Source Face Recognition Performance Evaluation Package

论文链接：https://arxiv.org/abs/1901.09447
 
摘要: Biometrics-related research has been accelerated significantly by deep learning technology. However, there are limited open-source resources to help researchers evaluate their deep learning-based biometrics algorithms efficiently, especially for the face recognition tasks. In this work, we design and implement a light-weight, maintainable, scalable, generalizable, and extendable face recognition evaluation toolbox named FaRE that supports both online and offline evaluation to provide feedback to algorithm development and accelerate biometrics-related research. FaRE consists of a set of evaluation metric functions and provides various APIs for commonly-used face recognition datasets including LFW, CFP, UHDB31, and IJB-series datasets, which can be easily extended to include other customized datasets. The package and the pre-trained baseline models will be released for public academic research use after obtaining university approval.

[6] 

论文题目：Convolutional Neural Networks with Layer Reuse

论文链接：https://arxiv.org/abs/1901.09615

摘要: A convolutional layer in a Convolutional Neural Network (CNN) consists of many filters which apply convolution operation to the input, capture some special patterns and pass the result to the next layer. If the same patterns also occur at the deeper layers of the network, why wouldn't the same convolutional filters be used also in those layers? In this paper, we propose a CNN architecture, Layer Reuse Network (LruNet), where the convolutional layers are used repeatedly without the need of introducing new layers to get a better performance. This approach introduces several advantages: (i) Considerable amount of parameters are saved since we are reusing the layers instead of introducing new layers, (ii) the Memory Access Cost (MAC) can be reduced since reused layer parameters can be fetched only once, (iii) the number of nonlinearities increases with layer reuse, and (iv) reused layers get gradient updates from multiple parts of the network. The proposed approach is evaluated on CIFAR-10, CIFAR-100 and Fashion-MNIST datasets for image classification task, and layer reuse improves the performance by 5.14%, 5.85% and 2.29%, respectively. The source code and pretrained models are publicly available.

[7] CollaGAN

论文题目：CollaGAN : Collaborative GAN for Missing Image Data Imputation

论文链接：https://arxiv.org/abs/1901.09764

摘要: In many applications requiring multiple inputs to obtain a desired output, if any of the input data is missing, it often introduces large amounts of bias. Although many techniques have been developed for imputing missing data, the image imputation is still difficult due to complicated nature of natural images. To address this problem, here we proposed a novel framework for missing image data imputation, called Collaborative Generative Adversarial Network (CollaGAN). CollaGAN converts an image imputation problem to a multi-domain images-to-image translation task so that a single generator and discriminator network can successfully estimate the missing data using the remaining clean data set. We demonstrate that CollaGAN produces the images with a higher visual quality compared to the existing competing approaches in various image imputation tasks.

[8] FG 2019 Sketch Generation文章

论文题目：Attribute-Guided Sketch Generation

论文链接：https://arxiv.org/abs/1901.09774

摘要: Facial attributes are important since they provide a detailed description and determine the visual appearance of human faces. In this paper, we aim at converting a face image to a sketch while simultaneously generating facial attributes. To this end, we propose a novel Attribute-Guided Sketch Generative Adversarial Network (ASGAN) which is an end-to-end framework and contains two pairs of generators and discriminators, one of which is used to generate faces with attributes while the other one is employed for image-to-sketch translation. The two generators form a W-shaped network (W-net) and they are trained jointly with a weight-sharing constraint. Additionally, we also propose two novel discriminators, the residual one focusing on attribute generation and the triplex one helping to generate realistic looking sketches. To validate our model, we have created a new large dataset with 8,804 images, named the Attribute Face Photo & Sketch (AFPS) dataset which is the first dataset containing attributes associated to face sketch images. The experimental results demonstrate that the proposed network (i) generates more photo-realistic faces with sharper facial attributes than baselines and (ii) has good generalization capability on different generative tasks.

[9] vcGAN

论文题目：Virtual Conditional Generative Adversarial Networks

论文链接：https://arxiv.org/abs/1901.09822

代码链接：https://github.com/annonnymmouss/vcgan

摘要: When trained on multimodal image datasets, normal Generative Adversarial Networks (GANs) are usually outperformed by class-conditional GANs and ensemble GANs, but conditional GANs is restricted to labeled datasets and ensemble GANs lack efficiency. We propose a novel GAN variant called virtual conditional GAN (vcGAN) which is not only an ensemble GAN with multiple generative paths while adding almost zero network parameters, but also a conditional GAN that can be trained on unlabeled datasets without explicit clustering steps or objectives other than the adversary loss. Inside the vcGAN's generator, a learnable ``analog-to-digital converter (ADC)" module maps a slice of the inputted multivariate Gaussian noise to discrete/digital noise (virtual label), according to which a selector selects the corresponding generative path to produce the sample. All the generative paths share the same decoder network while in each path the decoder network is fed with a concatenation of a different pre-computed amplified one-hot vector and the inputted Gaussian noise. We conducted a lot of experiments on several balanced/imbalanced image datasets to demonstrate that vcGAN converges faster and achieves improved Frechét Inception Distance (FID). In addition, we show the training byproduct that the ADC in vcGAN learned the categorical probability of each mode and that each generative path generates samples of specific mode, which enables class-conditional sampling.


[10] 

论文题目：CoCoNet: A Collaborative Convolutional Network

论文链接：https://arxiv.org/abs/1901.09886

摘要: We present an end-to-end CNN architecture for fine-grained visual recognition called Collaborative Convolutional Network (CoCoNet). The network uses a collaborative filter after the convolutional layers to represent an image as an optimal weighted collaboration of features learned from training samples as a whole rather than one at a time. This gives CoCoNet more power to encode the fine-grained nature of the data with limited samples in an end-to-end fashion. We perform a detailed study of the performance with 1-stage and 2-stage transfer learning and different configurations with benchmark architectures like AlexNet and VggNet. The ablation study shows that the proposed method outperforms its constituent parts considerably and consistently. CoCoNet also outperforms the baseline popular deep learning based fine-grained recognition method, namely Bilinear-CNN (BCNN) with statistical significance. Experiments have been performed on the fine-grained species recognition problem, but the method is general enough to be applied to other similar tasks. Lastly, we also introduce a new public dataset for fine-grained species recognition, that of Indian endemic birds and have reported initial results on it. The training metadata and new dataset are available through the corresponding author.


[11] ICLR 2019 Fixup Initialization文章

论文题目：Fixup Initialization: Residual Learning Without Normalization

论文链接：https://arxiv.org/abs/1901.09321

摘要: Normalization layers are a staple in state-of-the-art deep neural network architectures. They are widely believed to stabilize training, enable higher learning rate, accelerate convergence and improve generalization, though the reason for their effectiveness is still an active research topic. In this work, we challenge the commonly-held beliefs by showing that none of the perceived benefits is unique to normalization. Specifically, we propose fixed-update initialization (Fixup), an initialization motivated by solving the exploding and vanishing gradient problem at the beginning of training via properly rescaling a standard initialization. We find training residual networks with Fixup to be as stable as training with normalization -- even for networks with 10,000 layers. Furthermore, with proper regularization, Fixup enables residual networks without normalization to achieve state-of-the-art performance in image classification and machine translation.

[12] derain文章

论文题目：Progressive Image Deraining Networks: A Better and Simpler Baseline

论文链接：https://arxiv.org/abs/1901.09221

代码地址：https://github.com/csdwren/PReNet

摘要: Along with the deraining performance improvement of deep networks, their structures and learning become more and more complicated and diverse, making it difficult to analyze the contribution of various network modules when developing new deraining networks. To handle this issue, this paper provides a better and simpler baseline deraining network by considering network architecture, input and output, and loss functions. Specifically, by repeatedly unfolding a shallow ResNet, progressive ResNet (PRN) is proposed to take advantage of recursive computation. A recurrent layer is further introduced to exploit the dependencies of deep features across stages, forming our progressive recurrent network (PReNet). Furthermore, intra-stage recursive computation of ResNet can be adopted in PRN and PReNet to notably reduce network parameters with graceful degradation in deraining performance. For network input and output, we take both stage-wise result and original rainy image as input to each ResNet and finally output the prediction of {residual image}. As for loss functions, single MSE or negative SSIM losses are sufficient to train PRN and PReNet. Experiments show that PRN and PReNet perform favorably on both synthetic and real rainy images. Considering its simplicity, efficiency and effectiveness, our models are expected to serve as a suitable baseline in future deraining research.

[13] 

论文题目：4D Generic Video Object Proposals

论文链接：https://arxiv.org/abs/1901.09260

摘要: Many high-level video understanding methods require input in the form of object proposals. Currently, such proposals are predominantly generated with the help of networks that were trained for detecting and segmenting a set of known object classes, which limits their applicability to cases where all objects of interest are represented in the training set. This is a restriction for automotive scenarios, where unknown objects can frequently occur. We propose an approach that can reliably extract spatio-temporal object proposals for both known and unknown object categories from stereo video. Our 4D Generic Video Tubes (4D-GVT) method leverages motion cues, stereo data, and object instance segmentation to compute a compact set of video-object proposals that precisely localizes object candidates and their contours in 3D space and time. We show that given only a small amount of labeled data, our 4D-GVT proposal generator generalizes well to real-world scenarios, in which unknown categories appear. It outperforms other approaches that try to detect as many objects as possible by increasing the number of classes in the training set to several thousand.

[14] VISAPP 2019 Autonomous Driving数据集文章

论文题目：Challenges in Designing Datasets and Validation for Autonomous Driving

论文链接：https://arxiv.org/abs/1901.09270

摘要: Autonomous driving is getting a lot of attention in the last decade and will be the hot topic at least until the first successful certification of a car with Level 5 autonomy. There are many public datasets in the academic community. However, they are far away from what a robust industrial production system needs. There is a large gap between academic and industrial setting and a substantial way from a research prototype, built on public datasets, to a deployable solution which is a challenging task. In this paper, we focus on bad practices that often happen in the autonomous driving from an industrial deployment perspective. Data design deserves at least the same amount of attention as the model design. There is very little attention paid to these issues in the scientific community, and we hope this paper encourages better formalization of dataset design. More specifically, we focus on the datasets design and validation scheme for autonomous driving, where we would like to highlight the common problems, wrong assumptions, and steps towards avoiding them, as well as some open problems.


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






