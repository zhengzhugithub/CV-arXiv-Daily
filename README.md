# CV-arXiv-Daily

**分享计算机视觉每天的arXiv文章，主要集中在目标检测，单目标跟踪，多目标跟踪，人体行为识别，人体姿态估计与跟踪，行人重识别，模型搜索等。每周周末会将本周的Archive起来**

[2019-01-23~2019-01-26](2019/2019.01.23-2019.01.26.md)

[2019-01-28~2019-02-01](2019/2019.01.28-2019.02.01.md)

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










