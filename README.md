# CV-arXiv-Daily

**分享计算机视觉每天的arXiv文章，主要集中在目标检测，单目标跟踪，多目标跟踪，人体行为识别，人体姿态估计与跟踪，行人重识别，模型搜索等。每周周末会将本周的Archive起来**

[2019-01-23~2019-01-26](2019/2019.01.23-2019.01.26.md)

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






