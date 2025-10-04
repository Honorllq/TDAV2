# COSMIC: Clique-Oriented Semantic Multi-space Integration for Robust CLIP Test-Time Adaptation

Fanding Huang $^{1,*}$ , Jingyan Jiang $^{2,*}$ , Qinting Jiang $^{1}$ , Hebei Li $^{3}$ , Faisal Nadeem Khan $^{1,\dagger}$ , Zhi Wang $^{1,\dagger}$

$^{1}$ Shenzhen International Graduate School, Tsinghua University  $^{2}$ Shenzhen Technology University  $^{3}$ University of Science and Technology of China

# Abstract

Recent vision-language models (VLMs) face significant challenges in test-time adaptation to novel domains. While cache-based methods show promise by leveraging historical information, they struggle with both caching unreliable feature-label pairs and indiscriminately using single-class information during querying, significantly compromising adaptation accuracy. To address these limitations, we propose COSMIC (Clique-Oriented Semantic Multi-space Integration for CLIP), a robust test-time adaptation framework that enhances adaptability through multi-granular, cross-modal semantic caching and graph-based querying mechanisms. Our framework introduces two key innovations: Dual Semantics Graph (DSG) and Clique Guided Hyper-class (CGH). The Dual Semantics Graph constructs complementary semantic spaces by incorporating textual features, coarse-grained CLIP features, and fine-grained DINOv2 features to capture rich semantic relationships. Building upon these dual graphs, the Clique Guided Hyper-class component leverages structured class relationships to enhance prediction robustness through correlated class selection. Extensive experiments demonstrate COSMIC's superior performance across multiple benchmarks, achieving significant improvements over state-of-the-art methods:  $15.81\%$  gain on out-of-distribution tasks and  $5.33\%$  on cross-domain generation with CLIP RN-50. Code is available at github.com/ht618/COSMIC.

# 1. Introduction

Vision-language models (VLMs), such as CLIP [28] and ALIGN [17], have demonstrated remarkable performance across various downstream tasks, including semantic segmentation [21, 36] and video understanding [32, 35]. This success can be attributed primarily to the alignment of vi

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/12a6d7507cdb2375467242d39bab014019e7cb8959928fc046c699fa04876385.jpg)  
(a) Cache-based methods

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/3f3e33591fa526260aebb4fa8ed014b93353fc9c4bf26df43d5037e447be10c0.jpg)  
(b) Clique-Oriented Semantic Multi-space Integration for CLIP (Ours)  
Figure 1. (a) In the conventional cache-based method, the cache has only dull information with coarse-grained clip visual features and simple query way via similarity between samples and cached visual class centers. (b) In our COSMIC, the cache has diverse structural information via extra fine-grained DINOv2 visual features and effective query way via similarity between samples and meticulously designed hyper-class centers.

sual and textual features on large-scale datasets, enabling these models to exhibit robust image understanding capabilities in open-world scenarios. However, when deployed in real-world applications, these models often encounter test samples that significantly deviate from the training dataset, resulting in performance degradation [42, 43].

Recently, researchers [30] have explored the test-time adaptation (TTA) scenario for CLIP inference. TTA focuses on adapting models solely during testing by fine-tuning a small subset of the model's parameters, such as prompts (i.e., prompt learning), or even employing training-free cache-based methods to enhance the zero-shot image classification of CLIP. Prompt learning methods [1, 8, 30] optimize textual learnable prompts by minimizing entropy with confidence selection, ensuring consistent predictions across various augmented views of each test image. Using

the updated prompt, it generates adapted textual class features, making predictions based on their similarity to sample features. While effective, these approaches suffer from computational inefficiency due to numerous visual augmentations and iterative backpropagation steps.

In contrast, as shown in Fig. 1 (a), cache-based methods [18, 41] enhance model performance by utilizing historical information. Specifically, they propose an extra cache to store visual features with pseudo-labels to generate visual class centers which are the average of previous visual features of each class. When querying the cache, features of the new test image are compared to find similar class centers. Then, labels corresponding to those centers are chosen to provide more information for the final prediction. Due to its training-free design and ability to leverage global historical information, the cache-based approach surpasses prompt learning in both effectiveness and efficiency.

However, two issues remain unaddressed in previous research: (1) Noisy pseudo-labels during cache construction contaminate the cache with unreliable feature-label pairs; (2) During querying, each class center encapsulates only single-class information, leading to blind propagation of a single pseudo-label regardless of its reliability. These dual flaws in cache construction and query mechanisms critically undermine adaptation accuracy.

To address the aforementioned issues, we introduce COSMIC (Clique-Oriented Semantic Multi-space Integration for CLIP), a robust CLIP test-time adaptation framework. Our key idea is to leverage multi-granularity and cross-modal semantic information to enrich the semantic content in the cache with limited samples while utilizing graph structures to organize and query the cache robustly. Specifically, as shown in Fig. 1 (b), we design two core components: (1) Dual Semantics Graph (DSG) can enhance the semantic diversity of pseudo-labels by incorporating fine-grained DINOv2 visual features and cross-modal text features. Additionally, it bridges the gap between textual semantics and more granular visual semantics, thereby reducing the proportion of noisy feature and pseudo-label pairs in the cache. (2) Clique Guided Hyper-class (CGH) connects class centers and merges pseudo-labels to include information from multiple categories, thus improving the probability of containing correct label information in retrieved pseudo-labels. This, in turn, allows for selective use of this information in subsequent prediction phases, improving the model's robustness and accuracy.

For the Dual Semantics Graph, we initially construct complementary feature spaces with varying semantic granularity: (1) The CLIP Shared Semantic space unifies text embeddings and visual class centers calculated by historical test features. (2) The Auxiliary Fine-grained Visual space incorporates fine-grained visual class centers from self-supervised DINOv2 [26]. As shown in Fig. 2 (a),

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/5cf21041ee9c90107151297db782b980020fbc6e2e95ee123dbdaaea0319aea3.jpg)  
(a) Different Visual Affinity.

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/6bb1e9396536d7b40abb4d6c22fa29d27d2ea62b213e13aa5fb72ec2fe861365.jpg)  
Figure 2. (a) For various types of dogs, like "shiba inu" and "great Pyrenees" DINOv2 features offer more refined similarity perception than CLIP. (b) Samples in a single Gaussian cluster come from the same class, while the maximal clique cluster can explicitly represent cross-class correlations.  
(b) Different Feature Cluster.

CLIP's visual discrimination is less refined than DINOv2's for subtly varied pet images. To capture intricate feature dynamics, we transform dual Euclidean spaces into dual graph spaces, modeling nonlinear interactions between classes.

For the Clique Guided Hyper-class, we design hyperclasses within each graph to represent latent semantics from various classes. Unlike previous approaches using Gaussian distributions [9], as shown in Fig. 2 (b), our graph-based feature modeling facilitates robust relationships with fewer samples and effectively clusters samples from diverse classes. Therefore, we can efficiently search for maximal cliques to form the Clique Guided Hyper-class, which represents the centroids of diverse, highly-affiliated class centers in the dual graph. Then test features query the hyper-class centers by similarity to select highly correlated classes, referred to as inlier classes. Based on the logits of these inlier classes, we generate adapted predictions for test-time adaptation, ensuring a more robust and accurate model performance.

In summary, our contributions are as follows:

- We introduce COSMIC, a training-free test-time adaptation method for CLIP that explicitly captures complementary semantic relationships.  
- To refine cache construction, we design a Dual Semantics Graph that integrates intra-modal, cross-modal, coarse-grained, and fine-grained semantics.  
- To query cache adaptively, we design Clique Guided Hyper-Class to represent class clusters with high affinity, enabling more robust querying of test samples.

# 2. Related Work

# 2.1. Vision-Language Models

Pre-trained on large and high-quality datasets, vision-language models like CLIP [28] achieve strong generalization by aligning textual and visual features via contrastive learning. For few-shot image classification, prompt learning improves adaptability by optimizing learnable tokens on the textual [42, 43] or viscal [19] encoders.

To avoid backpropagation costs, methods like TipAdapter [38], CaFo [39] and GraphAdapter [22] use vision adapters for VLM generalization. Although these methods enrich feature semantics in few-shot settings, we focus on

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/cf1b6e47ad789bcc7230cde5716845469c0f475cf4453943f6f2477f7c72a9b9.jpg)  
Figure 3. Overview of COSMIC. To refine cache with cross-modal, multi-granular class features, we construct Dual Semantics Graph with complementary semantics, incorporating both joint modalities and fine-grained visual information. To efficiently query the compatibility of diverse semantics, we propose novel Clique Guided Hyper-class to model different communities in the cache as the test domain evolves, enabling adaptive querying of test samples.

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/34571f79a081207e056dcbd40348d9d19ffd7f88b5c42fcc9274560ce752c192.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/acb2d1044629f78cb4253222a7768de1a9f5142a2723dbfe387a5d04fd6a3869.jpg)

zero-shot test-time adaptation, rigorously exploring cross-modal and multi-granular class centers to address the nonlinear relationships in new domains, better aligning with real-world deployment scenarios.

# 2.2. Test Time Adaptation

Test-time adaptation addresses distribution shifts between training and test data. Recent approaches for VLMs include: TPT [30] leveraging prompt tuning for view consistency; SwapPrompt [23] employing self-contrastive learning for rapid adaptation; and PromptAlign [1] aligning cross-domain statistical features.

In contrast, cache-based methods like TDA [18] and DMN [41] construct dynamic caches to retrieve adapted visual class centers during testing. While this method enables CLIP to generalize via historical samples, such caches rely solely on CLIP's coarse text-image aligned features—lacking vision-specific contrastive knowledge critical for similarity awareness, as demonstrated by DINOv2 [26]. Our work bridges this gap by enhancing cached features with textual and fine-grained visual semantics.

# 2.3. Pre-training of Visual Encoders

Modern vision systems predominantly rely on a two-stage paradigm: large-scale pretraining followed by task-specific fine-tuning. Despite its effectiveness, this approach poses limitations due to label dependency. Self-supervised learning mitigates this bottleneck by learning annotation-free representations, achieving state-of-the-art performance in semantic segmentation [14, 15] and depth estimation [16].

Transitioning from multimodal to view-consistent learning overcomes critical visual granularity limitations. While CLIP [28] excels in cross-modal alignment through vision-language pretraining, its text-centric objective compromises spatial details vital for dense prediction. DINOv2 [26] resolves this by learning augmentation-invariant features through self-distillation, preserving pixel-level nuances for annotation-free localization without textual guidance.

# 3. Method

# 3.1. Preliminaries

CLIP model consists of two parallel encoders: a text encoder  $F_{t}$  and a visual encoder  $F_{v}$ , which map features from different modalities into a shared embedding space. In a zero-shot  $K$  class classification task, given a test image  $\mathbf{x}_{\mathrm{test}}$ , the visual feature  $\mathbf{w}_{\mathbf{v}} = F_{v}(\mathbf{x}_{\mathrm{test}})$  serves as a query to the text features of all class descriptions. The probability that  $\mathbf{x}_{\mathrm{test}}$  belongs to the class  $k$  can then be expressed as

$$
p _ {k} ^ {\mathrm {z s}} \left(\boldsymbol {x} _ {\text {t e s t}}\right) = \frac {\exp \left(\cos \left(\boldsymbol {w} _ {\mathbf {v}} , f _ {\mathbf {t} _ {k}}\right) / \tau\right)}{\sum_ {i = 1} ^ {K} \exp \left(\cos \left(\boldsymbol {w} _ {\mathbf {v}} , f _ {\mathbf {t} _ {i}}\right) / \tau\right)}, \tag {1}
$$

where  $f_{\mathbf{t}_i}$  represents the textual class feature for the  $i^{th}$  class, and  $\tau$  denotes the temperature parameter in the softmax function.

Inspired by few-shot adaptation of CLIP method TipAdapter [38], TDA [18] is a typical cache-based method for TTA that utilizes a cache to store knowledge from high-confidence historical test samples. This cache operates as a queue-style dictionary, where the keys represent the pseudo-labels of the samples, and the values store the corresponding visual features. The decision to add a sample to the cache depends on its prediction entropy. Specifically, for a test sample  $x_{\mathrm{test}}$ , after undergoing random  $\mathcal{N}$  augmentations  $\mathcal{A}$  (including the original image), the marginal entropy of the sample predictions is calculated as:

$$
H \left(\boldsymbol {x} _ {\text {t e s t}}\right) = - \sum_ {i = 1} ^ {K} \tilde {p} _ {i} \left(\boldsymbol {x} _ {\text {t e s t}}\right) \log \tilde {p} _ {i} \left(\boldsymbol {x} _ {\text {t e s t}}\right), \tag {2}
$$

$$
\tilde {p} _ {i} \left(\boldsymbol {x} _ {\text {t e s t}}\right) = \frac {1}{\mathcal {N} _ {\max }} \sum_ {j = 1} ^ {\mathcal {N} _ {\max }} p _ {i} ^ {\mathrm {z s}} \left(\mathcal {A} _ {j} \left(\boldsymbol {x} _ {\text {t e s t}}\right)\right),
$$

where  $p_i^{\mathrm{zs}}(\mathcal{A}_j(\boldsymbol{x}_{\mathrm{test}}))$  represents probability of class  $i$  generated by the model for the  $j^{th}$  augmented view of the test image.  $N_{max}$  is the number of first  $R \times \mathcal{N}$  high-confidence view predictions, where  $R$  is the selection ratio. Based on

this, the logic for storing  $\pmb{w}_{\mathbf{v}}$  predicted as class  $i$  into cache set  $\mathbb{M}_i$  is as follows:

$$
\mathbb {M} _ {i} = \left\{ \begin{array}{l l} \mathbb {M} _ {i} \cup \boldsymbol {w} _ {\mathbf {v}}, & \text {i f} | \mathbb {M} _ {i} | <   L, \\ \mathbb {M} _ {i} \setminus \left\{f _ {\max } \right\} \cup \boldsymbol {w} _ {\mathbf {v}}, & \text {i f} | \mathbb {M} _ {i} | = L \text {a n d} \\ \mathbb {M} _ {i}, & H (\boldsymbol {x} _ {\text {t e s t}}) <   H _ {\max } (\mathbb {M} _ {i}), \\ \end{array} \right. \tag {3}
$$

Here,  $f_{\mathrm{max}}$  denotes the feature in cached features set  $\mathbb{M}_i$  with the highest entropy  $H_{\mathrm{max}}(\mathbb{M}_i)$ , and  $L$  denotes the maximum capacity of the cache. Using this cache, the adapted predictions are computed as follows:

$$
\mathbf {P} _ {\text {a d a p t e d}} \left(\boldsymbol {x} _ {\text {t e s t}}\right) = \varphi \left(\boldsymbol {w} _ {\mathbf {v}} \mathbf {M} ^ {\top}\right) \mathbf {L} _ {\text {p s e u d o}}, \tag {4}
$$

where  $\varphi (x) = \exp (-\alpha (1 - x))$  is an adaptation function,  $\mathbf{M}\in \mathbb{R}^{KL\times d}$  is the cached features matrix, and  $\mathbf{L}_{\mathrm{pseudo}}\in$ $\mathbb{R}^{KL\times K}$  is the one-hot pseudo-label matrix.

# 3.2. Dual Semantics Graph (DSG)

# 3.2.1. CLIP Shared Semantic (CSS) Space

CLIP aims to align text and visual features in a shared feature space. Leveraging this, we combine text features  $\mathbf{f}_{\mathbf{t}} = [f_{\mathbf{t}_{1}}f_{\mathbf{t}_{2}}\dots f_{\mathbf{t}_{K}}]^{\top}\in \mathbb{R}^{K\times d_{1}}$  with cached visual class centers  $\mathbf{f}_{\mathbf{v}} = [f_{\mathbf{v}_{1}}f_{\mathbf{v}_{2}}\dots f_{\mathbf{v}_{K}}]^{\top}\in \mathbb{R}^{K\times d_{1}}$  in the same Euclidean space, forming a unified feature set  $\mathbf{f_s} = \mathbf{f_t}\cup \mathbf{f_v} = [f_{\mathbf{s}_1}f_{\mathbf{s}_2}\dots f_{\mathbf{s}_{2K}}]^{\top}\in \mathbb{R}^{2K\times d_1}$ , termed the CLIP Shared Semantic space with  $d_{1}$  dimension. The text feature  $f_{\mathbf{t}_i}$  are CLIP embeddings of class  $i$  description, defined as:

$$
f _ {\mathbf {t} _ {i}} = F _ {t} \left(\operatorname {T e x t} _ {i}\right), \tag {5}
$$

where  $\text{Text}_i$  is the input text of class  $i$ . Inspired by DMN [41], we construct the visual center  $f_{\mathbf{v}_i}$  for class  $i$  using a weighted combination of visual features, determined by the cosine similarity between test and cached visual features, illustrated as:

$$
f _ {\mathbf {v} _ {i}} = \varphi \left(\boldsymbol {w} _ {\mathbf {v}} \mathbf {M} _ {i} ^ {\top}\right) \mathbf {M} _ {i}, \tag {6}
$$

where  $\mathbf{M}_i\in \mathbb{R}^{l_1\times d_1}$  represents the cached visual feature matrix from the  $i^{th}$  category with a capacity  $l_{1}$  of cache.

# 3.2.2. Auxiliary Fine-grained Visual (AFV) Space

While the CLIP Shared Semantic space leverages both text and visual features for alignment, the cache model eliminates the need for text features during inference. However, relying solely on the CLIP visual encoder under noisy pseudo-labels can lead to inaccurate similarity perception, especially for easily confused images. Therefore, we introduce an auxiliary visual feature branch using self-supervised encoders like DINOv2 [26] to generate more finer-grained visual features of  $\mathbf{x}_{\mathrm{test}}$ .

Analogously to the CLIP visual feature cache, we establish a DINOv2 visual feature cache using the same criteria as in Eq. 3. To average historical test information with fine-grained semantics, we compute the auxiliary visual center  $f_{\mathbf{v}_i}^{\mathrm{aux}}$  via the centroid of cached DINOv2 features for class  $i$ :

$$
\mathbf {f} _ {\mathbf {v}} ^ {\mathrm {a u x}} = \left[ \begin{array}{c c c c} f _ {\mathbf {v} _ {1}} ^ {\mathrm {a u x}} & f _ {\mathbf {v} _ {2}} ^ {\mathrm {a u x}} & \dots & f _ {\mathbf {v} _ {C}} ^ {\mathrm {a u x}} \end{array} \right] ^ {\top} \in \mathbb {R} ^ {K \times d _ {2}},
$$

$$
f _ {\mathbf {v} _ {i}} ^ {\text {a u x}} = \frac {1}{l _ {2}} \sum_ {j = 1} ^ {l _ {2}} \mathbf {M} _ {i} ^ {\text {a u x}} [ j,: ], \tag {7}
$$

where  $l_{2}$  and  $d_{2}$  are the capacity of the auxiliary cache and dimension of AFV space, respectively.  $\mathbf{M}_i^{\mathrm{aux}}\in \mathbb{R}^{l_2\times d_2}$  is the auxiliary cached visual feature matrix for the  $i^{th}$  class.

# 3.2.3. Second-Order Graph Construction

In contrast to Euclidean space, graph space provides a more precise representation of complex affinity relationships [37, 40] between intra- and inter-modal features. Specifically, we construct two types of graphs: the First-Order Graph (FOG), which captures direct pairwise similarities, and the Second-Order Graph (SOG), which extends this by modeling higher-order relationships through the product of adjacency matrices, enabling the discovery of more complex and hidden patterns. In each FOG, nodes represent class center features, and edges capture bidirectional compatibility. Let  $\mathbf{F} \in \mathbb{R}^{N \times D}$  represent the feature matrix, where  $N = 2K$ ,  $D = d_{1}$  for CSS space and  $N = K$ ,  $D = d_{2}$  for AFV space. The adjacency matrix of First-Order Graph is defined as:

$$
\mathcal {W} _ {F O G} = \left[ w _ {i j} \right] _ {N \times N}, w _ {i j} = \left\{ \begin{array}{l l} 1 & \text {i f} \frac {\mathbf {F} _ {i} \cdot \mathbf {F} _ {j}}{\| \mathbf {F} _ {i} \| \| \mathbf {F} _ {j} \|} > t ^ {\text {a f f}}, \\ 0 & \text {o t h e r w i s e} \end{array} \right. \tag {8}
$$

where  $t^{\mathrm{aff}}$  is a threshold changed by test iterations. Overly concentrated class centers in specific domains can lead to excessive redundancy in graph information. To address this, the Second-Order Graph (SOG) captures higher-order dependencies through the adjacency matrix:

$$
\mathcal {W} _ {S O G} = \mathcal {W} _ {F O G} \odot \left(\mathcal {W} _ {F O G} \times \mathcal {W} _ {F O G}\right). \tag {9}
$$

Unlike FOG, SOG models indirect relationships, such as transitive or multi-hop connections, providing a more comprehensive view of node interactions. Additionally, the SOG is sparser than the FOG, reducing noise and improving computational efficiency. As testing progresses, the CSS and AFV graphs accumulate higher-quality features from new domains. To adapt to this dynamic, we adopt an increasing rule for the threshold  $t_i^{\mathrm{aff}} = \min \left(1, t_0^{\mathrm{aff}} + g \cdot i\right)$  where  $t_0^{\mathrm{aff}}$  is the initial value,  $i$  is the current number of samples tested, and  $g$  is a constant growth rate.

# 3.3. Clique Guided Hyper-class (CGH)

Maximal Cliques Search. Given an undirected graph  $\mathcal{G} = (\mathcal{V},\mathcal{E})$ , clique  $\mathcal{C} = (\mathcal{V}',\mathcal{E}')$ ,  $\mathcal{V}' \subseteq \mathcal{V}$ ,  $\mathcal{E}' \subseteq \mathcal{E}$  is a com

plete subgraph of  $\mathcal{G}$ . A maximal clique is a clique that cannot be extended by including any additional node. We employ the modified Bron-Kerbosch algorithm [6] for maximal clique search, which is a highly efficient algorithm with a worst-case time complexity of  $\mathcal{O}(b(n - b)3^{(b / 3)})$ , where  $b$  denotes the graph's degeneracy. When the  $t^{\mathrm{aff}}$  is high, the graph becomes sparser, leading to a lower degeneracy, which further accelerates the search process. For CSS and AFV spaces, we define their graphs:

$$
\mathcal {G} _ {\mathrm {C S S}} = \left(\mathcal {V} _ {\mathrm {C S S}}, \mathcal {E} _ {\mathrm {C S S}}\right), \mathcal {G} _ {\mathrm {A F V}} = \left(\mathcal {V} _ {\mathrm {A F V}}, \mathcal {E} _ {\mathrm {A F V}}\right), \tag {10}
$$

where  $\mathcal{G}_{\mathrm{CSS}}$  and  $\mathcal{G}_{\mathrm{AFV}}$  represent the graphs in the CSS and AFV spaces, respectively. For each graph  $j \in \{\mathrm{CSS}, \mathrm{AFV}\}$ , we apply the modified Bron-Kerbosch algorithm to both graphs:

$$
\mathbb {C} _ {j} = \operatorname {B r o n K e r b o s c h} (\mathcal {G} _ {j}), \tag {11}
$$

where  $\mathbb{C}_j = \{\mathcal{C}_{j1},\mathcal{C}_{j2},\dots,\mathcal{C}_{jm_j}\}$  is the set of maximal cliques in graph  $\mathcal{G}_j$ . These two sets of maximal cliques,  $\mathbb{C}_{\mathrm{CSS}}$  and  $\mathbb{C}_{\mathrm{AFV}}$ , enable us to capture a more comprehensive understanding of feature affinities across diverse levels of semantics and granularities.

Hyper-classes Generation. Building on the maximal cliques identified in both  $\mathcal{G}_{\mathrm{CSS}}$  and  $\mathcal{G}_{\mathrm{AFV}}$ , we further reveal latent patterns by searching for Clique-Guided Hyperclasses. Given the dense connectivity within maximal cliques, we define hyper-class centers as the centroids of node classes within clique  $i$ .

$$
f _ {j i} ^ {\text {h y p e r - c l a s s}} = \frac {1}{\left| \mathcal {C} _ {j i} \right|} \sum_ {f \in \mathcal {C} _ {j i}} f, \tag {12}
$$

where  $f$  represents class nodes in the respective graph. This approach models community structures between class centers through graph-based contextual reasoning, overcoming the locality bias of similarity-query paradigms. The affinity between test image feature  $w_{v}$  and clique  $i$  in  $\mathcal{G}_j$  is:

$$
\rho_ {j} \left(\boldsymbol {w} _ {\mathbf {v}}, \mathcal {C} _ {j i}\right) = \cos \left(\boldsymbol {w} _ {\mathbf {v}}, f _ {j i} ^ {\text {h y p e r - c l a s s}}\right). \tag {13}
$$

To identify the hyper-classes most proximal to the test sample, we sort the cliques by affinity:  $\rho_{j}(\pmb{w}_{\mathbf{v}},\mathcal{C}_{j(1)})\geq$ $\rho_{j}(\pmb{w}_{\mathbf{v}},\mathcal{C}_{j(2)})\geq \dots \geq \rho_{j}(\pmb{w}_{\mathbf{v}},\mathcal{C}_{j(m_{j})})$  . Then we select the top  $r$  proportion of the closest hyper-classes in each graph:

$$
\mathbb {C} _ {j} ^ {\text {s e l e c t e d}} = \left\{\mathcal {C} _ {j (1)}, \mathcal {C} _ {j (2)}, \dots , \mathcal {C} _ {j \left(k _ {j}\right)} \right\}, \tag {14}
$$

where  $k_{j} = \lceil r\cdot m_{j}\rceil$ . This procedure generates two hyperclass guided masks  $\mathcal{M}_j$ , effectively delineating the selected classes as inliers while designating the remainder as outliers (classes with low relevance to the test sample).

$$
\begin{array}{l} \mathcal {M} _ {j} = \left[ \begin{array}{c c c c} m _ {j 1} & m _ {j 2} & \ldots & m _ {j N _ {j}} \end{array} \right] \in \mathbb {R} ^ {1 \times N _ {j}}, \\ m _ {j i} = \left\{ \begin{array}{l l} 1, & \text {i f} i \in \bigcup_ {\mathcal {C} \in \mathbb {C} _ {j} ^ {\text {s e l e c t e d}}} \mathcal {C} \\ 0, & \text {o t h e r w i s e} \end{array} , \right. \tag {15} \\ \end{array}
$$

where  $N_{\mathrm{CSS}} = 2K$  for the CLIP Shared Semantic space and  $N_{\mathrm{AFV}} = K$  for the Auxiliary Fine-grained Visual space.

# 3.4. Adaptive Inference with Hyper-class

The  $\mathcal{M}_{\mathrm{CSS}}$  is then applied to the initial logits of all classes in the CSS space:

$$
\mathbf {P} _ {\mathrm {C S S}} ^ {\text {i n i t i a l}} = \operatorname {S o f t m a x} \left(\boldsymbol {w} _ {\mathbf {v}} \mathbf {f} _ {\mathbf {s}} ^ {\top}\right) \odot \mathcal {M} _ {\mathrm {C S S}} \in \mathbb {R} ^ {1 \times 2 K}. \tag {16}
$$

The probability  $p_i$  for the  $i^{th}$  class in the CSS space is calculated as the average of the text and image class predictions:

$$
p _ {i} = \left(p _ {i} ^ {\text {i n i t i a l}} + p _ {i + K} ^ {\text {i n i t i a l}}\right) / 2, \quad \text {f o r} \quad i = 1, 2, \dots , K. \tag {17}
$$

Let  $p_i^{initial}$  denote the prediction value in the  $i^{th}$  column of  $\mathbf{P}_{\mathrm{CSS}}^{initial}$ . The classification probability distribution in the CSS space is then given by:

$$
\mathbf {P} _ {\mathrm {C S S}} = \left[ \begin{array}{l l l l} p _ {1} & p _ {2} & \dots & p _ {K} \end{array} \right] \in \mathbb {R} ^ {1 \times K}, \tag {18}
$$

where  $K$  is the number of classes. Following a similar process, we obtain the class prediction probability distribution within the AFV space:

$$
\mathbf {P} _ {\mathrm {A F V}} = \operatorname {S o f t m a x} \left(\boldsymbol {w} _ {\mathbf {v}} ^ {\text {a u x}} \mathbf {f} _ {\mathbf {v}} ^ {\text {a u x} ^ {\top}}\right) \odot \mathcal {M} _ {\mathrm {A F V}} \in \mathbb {R} ^ {1 \times K}, \tag {19}
$$

where  $\boldsymbol{w}_{\mathbf{v}}^{\mathrm{aux}}$  is the auxiliary image feature of test image  $\boldsymbol{x}_{\mathrm{test}}$ . The calculation of the  $\mathcal{M}_{\mathrm{AFV}}$  is similar to  $\mathcal{M}_{\mathrm{CSS}}$ , but they mainly filter out outliers from various feature clusters, thereby preventing prediction bias caused by the accumulation of moderate logits values. We combine initial CLIP prediction with adaptive prediction into the final prediction:

$$
\mathbf {P} _ {\text {F i n a l}} = \beta_ {1} \mathbf {P} _ {\mathrm {Z S}} + \beta_ {2} \mathbf {P} _ {\mathrm {C S S}} + \beta_ {3} \mathbf {P} _ {\mathrm {A F V}}, \tag {20}
$$

$$
\mathbf {P} _ {\mathrm {Z S}} = \operatorname {S o f t m a x} \left(\boldsymbol {w} _ {\mathbf {v}} \mathbf {f} _ {\mathbf {t}}\right) ^ {\top} \in \mathbb {R} ^ {1 \times K},
$$

where  $\beta_{1},\beta_{2},\beta_{3}$  represent the weights, and their values will be discussed in detail in Sec. 4.3.3.

# 4. Experiments

# 4.1. Experimental Settings

Datasets. In the VLM test-time adaptation setting, two main benchmark types are typically used. The first evaluates the model's robustness under out-of-distribution (OOD) shifts, while the second examines cross-domain (CD) generalization capabilities.

- For OOD shifts, we employ the ImageNet validation [4] along with four variants: ImageNet-A [13], ImageNet-V2 [29], ImageNet-R [12], and ImageNet-Sketch [33].  
- For CD tasks, we utilize ten diverse sub-datasets, each representing a distinct domain: Aircraft [24], Caltech101 [7], Cars [20], DTD [3], EuroSAT [11], Flower102 [25], Food101 [2], Pets [27], SUN397 [34], and UCF101 [31].

Table 1. Top-1 accuracy (%) comparison on ImageNet and its OOD variants using CLIP with ResNet-50 and ViT-B/16 backbones. Our results are reported as mean±std over 3 random seeds. Bold indicates the highest performance.  

<table><tr><td>Method</td><td>Adaptation Settings</td><td>ImageNet</td><td>ImageNet-A</td><td>ImageNet-V2</td><td>ImageNet-R</td><td>ImageNet-S</td><td>Average</td><td>OOD Average</td></tr><tr><td>CLIP-RN-50</td><td>-</td><td>58.16</td><td>21.83</td><td>51.41</td><td>56.15</td><td>33.37</td><td>44.18</td><td>40.69</td></tr><tr><td>CoOp [43] (IJCV&#x27;22)</td><td>Training Few-shot</td><td>63.33</td><td>23.06</td><td>55.40</td><td>56.60</td><td>34.67</td><td>46.61</td><td>42.43</td></tr><tr><td>CoCoOp [42] (CVPR&#x27;22)</td><td>Training Few-shot</td><td>62.81</td><td>23.32</td><td>55.72</td><td>57.74</td><td>34.48</td><td>46.81</td><td>42.82</td></tr><tr><td>TPT [30] (NeurIPS&#x27;22)</td><td>Training Zero-shot</td><td>60.74</td><td>26.67</td><td>54.70</td><td>59.11</td><td>35.09</td><td>47.26</td><td>43.89</td></tr><tr><td>DiffTPT [8] (ICCV&#x27;23)</td><td>Training Zero-shot</td><td>60.80</td><td>31.06</td><td>55.80</td><td>58.80</td><td>37.10</td><td>48.71</td><td>45.69</td></tr><tr><td>TDA [18] (CVPR&#x27;24)</td><td>Training-free Zero-shot</td><td>61.35</td><td>30.29</td><td>55.54</td><td>62.58</td><td>38.12</td><td>49.58</td><td>46.63</td></tr><tr><td>DMN [41] (CVPR&#x27;24)</td><td>Training-free Zero-shot</td><td>63.87</td><td>28.57</td><td>56.12</td><td>61.44</td><td>39.84</td><td>49.97</td><td>46.49</td></tr><tr><td>COSMIC (Ours)</td><td>Training-free Zero-shot</td><td>75.19±0.89</td><td>49.07±0.06</td><td>63.81±0.39</td><td>79.87±0.08</td><td>56.44±0.34</td><td>64.88±0.22</td><td>62.30±0.05</td></tr><tr><td>Method</td><td>Adaptation Settings</td><td>ImageNet</td><td>ImageNet-A</td><td>ImageNet-V2</td><td>ImageNet-R</td><td>ImageNet-S</td><td>Average</td><td>OOD Average</td></tr><tr><td>CLIP-ViT-B/16</td><td>-</td><td>66.73</td><td>47.87</td><td>60.86</td><td>73.98</td><td>46.09</td><td>59.11</td><td>57.20</td></tr><tr><td>CoOp [43] (IJCV&#x27;22)</td><td>Training Few-shot</td><td>71.51</td><td>49.71</td><td>64.20</td><td>75.21</td><td>47.99</td><td>61.72</td><td>59.28</td></tr><tr><td>CoCoOp [42] (CVPR&#x27;22)</td><td>Training Few-shot</td><td>71.02</td><td>50.63</td><td>64.07</td><td>76.18</td><td>48.75</td><td>62.13</td><td>59.91</td></tr><tr><td>TPT [30] (NeurIPS&#x27;22)</td><td>Training Zero-shot</td><td>68.98</td><td>54.77</td><td>63.45</td><td>77.06</td><td>47.94</td><td>62.44</td><td>60.81</td></tr><tr><td>DiffTPT [8] (ICCV&#x27;23)</td><td>Training Zero-shot</td><td>70.30</td><td>55.68</td><td>65.10</td><td>75.00</td><td>46.80</td><td>62.58</td><td>60.65</td></tr><tr><td>TDA [18] (CVPR&#x27;24)</td><td>Training-free Zero-shot</td><td>69.51</td><td>60.11</td><td>64.67</td><td>80.24</td><td>50.54</td><td>65.01</td><td>63.89</td></tr><tr><td>DMN [41] (CVPR&#x27;24)</td><td>Training-free Zero-shot</td><td>72.25</td><td>58.28</td><td>65.17</td><td>78.55</td><td>53.20</td><td>65.49</td><td>63.80</td></tr><tr><td>COSMIC (Ours)</td><td>Training-free Zero-shot</td><td>78.19±0.56</td><td>73.32±0.32</td><td>69.62±0.19</td><td>85.60±0.12</td><td>62.79±0.10</td><td>73.90±0.05</td><td>72.83±0.10</td></tr></table>

Implement Settings. We employed CLIP's officially pretrained vision encoders, including ResNet50 [10] and ViT-B/16 [5]. Inspired by previous works [41], we utilized handcrafted textual prompts and multi-view augmentations of the original images to calculate prediction confidence. For all experiments, we generated 16 views per image and fixed hyperparameters across each sub-dataset. To combine logits, we used an adaptive step search to determine optimal weights while also demonstrating the robustness of fixed weights. Unless otherwise specified, we employ DINOv2 ViT-L/14 as the auxiliary visual encoder. All experiments were conducted on a single Tesla V100S-PCIE-32GB GPU, using top-1 accuracy for classification performance.

Comparison Methods. We initially evaluate few-shot methods such as CoOp [43] and CoCoOp [42], using 16-shot annotated samples per class. Next, we assess training-based zero-shot methods. TPT [30] straightforwardly optimizes the text prompts by minimizing the multi-view marginal entropy. DiffTPT [8] is an advanced iteration of TPT, employing diffusion-based augmentations to refine prompts. Furthermore, we evaluate training-free zero-shot methods. TDA [18] is an adapter-based approach that builds positive and negative caches at test time without training. DMN [41] leverages a dynamic memory to compile information from past test data, bypassing backpropagation.

# 4.2. Comparison with SOTA

# 4.2.1. Results on the OOD Benchmark

In Tab. 1, we compare model performance on the in-domain ImageNet validation set and four OOD variants. Due to domain shifts, the zero-shot generalization ability of CLIP is limited in OOD scenarios. Notably, while CoOp [43] and CoCoOp [42] enhance CLIP's transferability by fine-tuning

prompts on few-shot samples, this incurs additional training costs, making it impractical for real-world deployment. In contrast, our proposed method is training-free, allowing for immediate adaptation to unseen domains. Furthermore, our method achieves substantial improvements over CoCoOp, with gains of  $19.48\%$  and  $12.92\%$  across different CLIP visual encoders, demonstrating its superior performance in OOD generalization tasks.

Additionally, our method outperforms previous test-time adaptation techniques. For prompt-learning-based methods, our approach surpasses TPT [30] and DiffTPT [8], with improvements of  $18.41\%$ ,  $12.02\%$ , and  $16.61\%$ ,  $12.18\%$  on two different backbones with OOD shifts, respectively, highlighting the effectiveness of our training-free approach. In comparison to cache model-based methods, our approach achieves superior results over the strongest techniques, TDA [18] and DMN [41], with gains of  $15.67\%$ ,  $15.81\%$ , and  $8.94\%$ ,  $9.03\%$  on two backbones with OOD shifts, respectively, validating the effectiveness of our Dual Semantics Graph and Clique Guided Hyper-class to improve the refinement and querying in cache.

# 4.2.2. Results on the CD Benchmark

We conducted a comparison on a diverse cross-domain dataset against a variety of contemporary methods, as shown in Tab. 2. Notably, both few-shot and zero-shot prompt learning methods show limited improvements in generalization performance. This limitation arises because, when there is a significant shift both in textual and visual modality, it becomes challenging to identify an optimal prompt within a constrained parameter space. By circumventing complex prompt space searches, our method adaptively bridges domain gaps through feature affinity mod

Table 2. Top-1 accuracy (%) comparison on 10 diverse cross-domain datasets using CLIP with ResNet-50 and ViT-B/16 backbones. Our results are reported as mean±std over 3 random seeds. Bold indicates the highest performance.  

<table><tr><td>Method</td><td>Adaptation Settings</td><td>Aircraft</td><td>Caltech101</td><td>Cars</td><td>DTD</td><td>EuroSAT</td><td>Flower102</td><td>Food101</td><td>Pets</td><td>SUN397</td><td>UCF101</td><td>Average</td></tr><tr><td>CLIP-RN-50</td><td>-</td><td>15.66</td><td>85.88</td><td>55.70</td><td>40.37</td><td>23.69</td><td>61.75</td><td>73.97</td><td>83.57</td><td>58.80</td><td>58.84</td><td>55.82</td></tr><tr><td>CoOp [43] (IICV&#x27;22)</td><td>Training Few-shot</td><td>15.12</td><td>86.53</td><td>55.32</td><td>37.29</td><td>26.20</td><td>61.55</td><td>75.59</td><td>87.00</td><td>58.15</td><td>59.05</td><td>56.18</td></tr><tr><td>CoCoOp [42] (CVPR&#x27;22)</td><td>Training Few-shot</td><td>14.61</td><td>87.38</td><td>56.22</td><td>38.53</td><td>28.73</td><td>65.57</td><td>76.20</td><td>88.39</td><td>59.61</td><td>57.10</td><td>57.23</td></tr><tr><td>TPT [30] (NeurIPS&#x27;22)</td><td>Training Zero-shot</td><td>17.58</td><td>87.02</td><td>58.46</td><td>40.84</td><td>28.33</td><td>62.69</td><td>74.88</td><td>84.49</td><td>61.46</td><td>60.82</td><td>57.66</td></tr><tr><td>DiffTPT [8] (ICCV&#x27;23)</td><td>Training Zero-shot</td><td>17.60</td><td>86.89</td><td>60.71</td><td>40.72</td><td>41.04</td><td>63.53</td><td>79.21</td><td>83.40</td><td>62.72</td><td>62.67</td><td>59.85</td></tr><tr><td>TDA [18] (CVPR&#x27;24)</td><td>Training-free Zero-shot</td><td>17.61</td><td>89.70</td><td>57.78</td><td>43.74</td><td>42.11</td><td>68.74</td><td>77.75</td><td>86.18</td><td>62.53</td><td>64.18</td><td>61.03</td></tr><tr><td>DMN [41] (CVPR&#x27;24)</td><td>Training-free Zero-shot</td><td>22.77</td><td>90.14</td><td>60.02</td><td>50.41</td><td>48.72</td><td>67.93</td><td>76.70</td><td>86.78</td><td>64.39</td><td>65.34</td><td>63.32</td></tr><tr><td>COSMIC (Ours)</td><td>Training-free Zero-shot</td><td>25.49±0.57</td><td>94.77±0.35</td><td>66.69±0.51</td><td>55.44±0.68</td><td>48.97±2.41</td><td>77.78±1.22</td><td>83.53±0.11</td><td>92.63±0.29</td><td>69.73±0.08</td><td>71.51±0.78</td><td>68.65±0.32</td></tr><tr><td>Method</td><td>Adaptation Settings</td><td>Aircraft</td><td>Caltech101</td><td>Cars</td><td>DTD</td><td>EuroSAT</td><td>Flower102</td><td>Food101</td><td>Pets</td><td>SUN397</td><td>UCF101</td><td>Average</td></tr><tr><td>CLIP-ViT-B/16</td><td>-</td><td>23.22</td><td>93.55</td><td>66.11</td><td>45.04</td><td>50.42</td><td>66.99</td><td>82.86</td><td>86.92</td><td>65.63</td><td>65.16</td><td>64.59</td></tr><tr><td>CoOp [43] (IICV&#x27;22)</td><td>Training Few-shot</td><td>18.47</td><td>93.70</td><td>64.51</td><td>41.92</td><td>46.39</td><td>68.71</td><td>85.30</td><td>89.14</td><td>64.15</td><td>66.55</td><td>63.88</td></tr><tr><td>CoCoOp [42] (CVPR&#x27;22)</td><td>Training Few-shot</td><td>22.29</td><td>93.79</td><td>64.90</td><td>45.45</td><td>39.23</td><td>70.85</td><td>83.97</td><td>90.46</td><td>66.89</td><td>68.44</td><td>64.63</td></tr><tr><td>TPT [30] (NeurIPS&#x27;22)</td><td>Training Zero-shot</td><td>24.78</td><td>94.16</td><td>66.87</td><td>47.75</td><td>42.44</td><td>68.98</td><td>84.67</td><td>87.79</td><td>65.50</td><td>68.04</td><td>65.10</td></tr><tr><td>DiffTPT [8] (ICCV&#x27;23)</td><td>Training Zero-shot</td><td>25.60</td><td>92.49</td><td>67.01</td><td>47.00</td><td>43.13</td><td>70.10</td><td>87.23</td><td>88.22</td><td>65.74</td><td>62.67</td><td>64.92</td></tr><tr><td>TDA [18] (CVPR&#x27;24)</td><td>Training-free Zero-shot</td><td>23.91</td><td>94.24</td><td>67.28</td><td>47.40</td><td>58.00</td><td>71.42</td><td>86.14</td><td>88.63</td><td>67.62</td><td>70.66</td><td>67.53</td></tr><tr><td>DMN [41] (CVPR&#x27;24)</td><td>Training-free Zero-shot</td><td>30.03</td><td>95.38</td><td>67.96</td><td>55.85</td><td>59.43</td><td>74.49</td><td>85.08</td><td>92.04</td><td>70.18</td><td>72.51</td><td>70.30</td></tr><tr><td>COSMIC (Ours)</td><td>Training-free Zero-shot</td><td>31.44±0.56</td><td>96.80±0.42</td><td>71.31±0.46</td><td>58.23±1.40</td><td>58.82±0.40</td><td>82.14±0.49</td><td>86.60±0.09</td><td>94.19±0.09</td><td>72.33±0.06</td><td>76.20±0.55</td><td>72.81±0.09</td></tr></table>

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/89c6609ccfdcfa1f90c0495c941df14819124dcc95854372b518a7ed9de680a7.jpg)  
Figure 4. Visualization of the attention maps from CLIP ViT-B/16 and DINOv2 ViT-B/14, along with their top-5 prediction results.

Table 3. Efficiency and effectiveness comparison of TTA methods on Flower102 [25] dataset using CLIP ViT-B/16.  

<table><tr><td>Method</td><td>Views</td><td>Time Per Image (s)</td><td>Top-1 Accuracy (%)</td><td>Gain (%)</td></tr><tr><td>CLIP [28]</td><td>1</td><td>0.098</td><td>66.99</td><td>-</td></tr><tr><td>TPT [30]</td><td>64</td><td>0.198</td><td>68.98</td><td>1.99</td></tr><tr><td>TDA [18]</td><td>64</td><td>0.119</td><td>71.42</td><td>4.43</td></tr><tr><td>DMN [41]</td><td>128</td><td>0.483</td><td>74.49</td><td>7.50</td></tr><tr><td>COSMIC</td><td>16</td><td>0.368</td><td>82.46</td><td>15.47</td></tr></table>

eling. For example, our approach achieves gains over TPT [30] of  $10.99\%$  and  $7.71\%$  across two CLIP visual backbones, respectively, validating its robustness.

Similarly, cache model-based methods typically adapt solely to the image distribution of the new domain. However, our method unifies query processing for both textual and cached image features, enabling it to adapt to cross-modal information distributions simultaneously. Our approach achieves improvements of  $7.62\%$  and  $5.28\%$  over TDA [18] across two backbones, further supporting the efficacy of our unified strategy.

# 4.2.3. Computation Efficiency

As shown in Tab. 3, we evaluated the efficiency of the Flowers dataset [25] on a single Tesla V100S-PCIE-32GB GPU. Note that the number of augmentation views is based on the specifications stated in the original papers for each method. By using a minimal number of image augmentations, we

Table 4. Graph ablation with CLIP ViT-B/16 and DINOv2 ViTS/14. CSS denotes the prediction from the CLIP Shared Semantic Graph, while AFV represents the prediction from the Auxiliary Fine-grained Visual Graph.  

<table><tr><td rowspan="2">CLIP</td><td rowspan="2">CSS</td><td rowspan="2">AFV</td><td colspan="2">ImageNet-Val [4]</td><td colspan="2">Flower102 [25]</td></tr><tr><td>Top-1 Accuracy (%)</td><td>Gain (%)</td><td>Top-1 Accuracy (%)</td><td>Gain (%)</td></tr><tr><td>✓</td><td></td><td></td><td>69.91</td><td>-</td><td>72.64</td><td>-</td></tr><tr><td></td><td>✓</td><td></td><td>71.88</td><td>1.97</td><td>74.46</td><td>1.82</td></tr><tr><td></td><td></td><td>✓</td><td>67.05</td><td>-2.86</td><td>77.71</td><td>5.07</td></tr><tr><td>✓</td><td>✓</td><td></td><td>72.06</td><td>2.15</td><td>75.07</td><td>2.43</td></tr><tr><td>✓</td><td></td><td>✓</td><td>73.16</td><td>3.25</td><td>79.74</td><td>7.10</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>73.79</td><td>3.88</td><td>81.61</td><td>8.97</td></tr></table>

effectively avoided excessive overhead while achieving a  $15.47\%$  improvement over the original CLIP. Additionally, our supplementary material includes analysis and experiments on accelerating inference, highlighting further optimization potential.

# 4.3. Ablation Study

# 4.3.1. Ablation of Graph Type

We evaluated the performance of different graphs on ImageNet Val [4] and Flowers102 [25]. As shown in the Tab. 4, when CSS and AFV prediction are fused with the original CLIP predictions, significant improvements are observed, with gains of  $2.15\%$ ,  $3.25\%$ , and  $2.43\%$ ,  $7.10\%$ , respectively. This indicates that each graph contributes to enhancing the generalization ability of CLIP. Moreover, the performance is optimized when CSS and AFV are used in

Table 5. Ablation of various auxiliary visual encoder in ImageNet Val [4] with CLIP ViT-B/16.  

<table><tr><td>DINOv2 Backbone</td><td>Extra Parameters</td><td>Top-1 Accuracy (%)</td><td>Gain (%)</td></tr><tr><td>-</td><td>-</td><td>69.91</td><td>-</td></tr><tr><td>ViT-S/14</td><td>21 M</td><td>73.79</td><td>3.88</td></tr><tr><td>ViT-B/14</td><td>86 M</td><td>76.73</td><td>6.82</td></tr><tr><td>ViT-L/14</td><td>300 M</td><td>77.54</td><td>7.63</td></tr></table>

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/5d0ffc3ea2dfb49363826a4e877ce6ada787766989c2850db805d7c402a37908.jpg)  
(a) ImageNet-V2 [29]  
Figure 5. Ablation of  $\beta_{2}$ ,  $\beta_{3}$  with CLIP ViT-B/16 and DINOv2 ViT-S/14.

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/7a20941473d62f30f9f7e5c9f01b822b77f5418215bac9ddb5157d4e1c8b5975.jpg)  
(b) Caltech101 [7]

conjunction, achieving improvements of  $3.88\%$  and  $8.97\%$ , which validates that the different target feature spaces enhance the perception of shared modal features and fine-grained visual capabilities. However, we note a  $-2.86\%$  degradation when using AFV alone on ImageNet-Val, as relying solely on cached DINOv2 visual features is insufficient to accurately describe new images with domain gaps.

# 4.3.2. Ablation of Auxiliary Visual Encoder

Introducing an auxiliary visual encoder with various sizes consistently improves model accuracy, as shown in Tab. 5. Specifically, each encoder boosts accuracy over the baseline (69.91%), with gains of  $3.67\%$  (ViT-S/14),  $6.82\%$  (ViT-B/14), and  $7.63\%$  (ViT-L/14). Furthermore, the Tab. 5 highlights a clear trade-off between performance gains and additional parameters. The smallest encoder, ViT-S/14, with only 21M extra parameters, shows modest improvement, while the largest, ViT-L/14, requires 300M additional parameters for the highest gain.

# 4.3.3. Ablation of Logits Weights

To verify the robustness of different predictive logits on the final prediction in Eq. 20, we fixed the value of  $\beta_{1}$  at 1 and set the step size to 0.05, varying  $\beta_{2}$  and  $\beta_{3}$  from 0 to 10 to observe their impact on model prediction accuracy. As shown in Fig. 5, we visualized this on two representative subsets ImageNet-V2 [29] and Caltech101 [7]. As shown, increasing either  $\beta_{2}$  or  $\beta_{3}$  significantly enhances accuracy, indicating that both semantic granularities independently improve generalization. These parameters exhibit strong synergistic properties, mutually promoting the fusion of complementary semantics and thereby achieving superior performance. The model demonstrates robustness,

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/eedefcf17f84079a3401e17b9480d0529a0ed3257c62a52af9567c914c974374.jpg)  
(a) Ablation of key modules.  
Figure 6. Ablation of components with CLIP-ViT-B16.

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/42c5fac3836adba0dd3402ac3052ecf93c367cfe5db07c5b054a24207475da15.jpg)  
(b) Ablation of  $r$

maintaining high accuracy even with perturbations around the optimal  $\beta$  values, suggesting that performance is resilient to variations in these hyperparameters.

# 4.3.4. Ablation of Components

In Fig. 6 (a),  $\mathrm{CLIP + DSG}$  is equivalent to directly computing the similarity between test samples and class nodes within the DSG. Although DSG independently enhances model generalization, CGH highlights the importance of selecting an appropriate  $r$  to enhance the model's robustness by effectively filtering out noise and outliers, thereby improving prediction accuracy. As shown in Fig. 6 (b), we varied the selection ratio  $r$  of nearest anchors across multiple datasets to evaluate the impact of different components on model performance. The results indicate that the model achieves optimal performance at  $r = 0.2$  with both DSG and CGH. This suggests that the similarity between test samples and diverse hyper-classes is more robust than that between samples and normal classes.

# 4.3.5. Visualization

As shown in Fig. 4, DINOV2's vision-contrastive encoder outperforms CLIP in capturing key objects. Leveraging finer-grained features, our method excels in recognizing "hard" classes, accurately distinguishing similar pets (e.g., Pomeranian vs. Japanese Chin) with higher confidence.

# 5. Conclusion

In CLIP test-time adaptation, we focus on exploring the potential of the cache-based method. As for how to refine cache, we introduced a Dual Semantics Graph to explore inter- and cross-modal affinities with various semantics. As for how to query cache, we introduce Clique Guided Hyper-classes within dual graphs to enhance the selection of correlated classes for robust predictions. Our method outperforms SOTA on multiple benchmarks, demonstrating strong zero-shot capabilities, while revealing the significant potential of structured semantic integration for robust visual understanding. Limitation: The current clique search methodology incurs non-trivial time complexity, imposing an additional computational burden on COSMIC. To address this, we aim to explore dual graph sparsification techniques for accelerated search in future iterations.

# Acknowledgment

This work was supported by the following funding sources: the National Key Research and Development Project of China (Grant No. 2023YFF0905502), the National Natural Science Foundation of China (Grant Nos. 92467204 and 62472249), the Shenzhen Science and Technology Program (Grant Nos. JCYJ20220818101014030 and KJZD20240903102300001), the Scientific Research Startup Fund of TSIGS (Project No. QD2022004C), the National Natural Science Foundation of China (Grant No. W2432041), and the Natural Science Foundation of Top Talent of SZTU (Grant No. GDRC202413).

# References

[1] Jameel Abdul Samadh, Mohammad Hanan Gani, Noor Hussein, Muhammad Uzair Khattak, Muhammad Muzammal Naseer, Fahad Shahbaz Khan, and Salman H Khan. Align your prompts: Test-time prompting with distribution alignment for zero-shot generalization. Advances in Neural Information Processing Systems, 36, 2024. 1, 3  
[2] Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101-mining discriminative components with random forests. In Computer vision-ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part VI 13, pages 446-461. Springer, 2014. 5, 4  
[3] Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3606-3613, 2014. 5, 4  
[4] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248-255. IEEE, 2009. 5, 7, 8, 1, 2, 4  
[5] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 6  
[6] David Eppstein, Maarten Löffler, and Darren Strash. Listing all maximal cliques in sparse graphs in near-optimal time. In Algorithms and Computation: 21st International Symposium, ISAAC 2010, Jeju Island, Korea, December 15-17, 2010, Proceedings, Part I 21, pages 403-414. Springer, 2010. 5  
[7] Li Fei-Fei, Rob Fergus, and Pietro Perona. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In 2004 conference on computer vision and pattern recognition workshop, pages 178-178. IEEE, 2004. 5, 8, 4  
[8] Chun-Mei Feng, Kai Yu, Yong Liu, Salman Khan, and Wangmeng Zuo. Diverse data augmentation with diffusions for effective test-time prompt tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2704-2714, 2023. 1, 6, 7  
[9] Zongbo Han, Jialong Yang, Junfan Li, Qinghua Hu, Qianli Xu, Mike Zheng Shou, and Changqing Zhang. Dota: Dis

tributional test-time adaptation of vision-language models. arXiv preprint arXiv:2409.19375, 2024. 2  
[10] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770-778, 2016. 6  
[11] Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7):2217-2226, 2019. 5, 4  
[12] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. The many faces of robustness: A critical analysis of out-of-distribution generalization. In Proceedings of the IEEE/CVF international conference on computer vision, pages 8340-8349, 2021. 5, 4  
[13] Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. Natural adversarial examples. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15262-15271, 2021. 5, 4  
[14] Lukas Hoyer, Dengxin Dai, and Luc Van Gool. Daformer: Improving network architectures and training strategies for domain-adaptive semantic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9924-9935, 2022. 3  
[15] Fanding Huang, Zihao Yao, and Wenhui Zhou. Dtbs: Dualteacher bi-directional self-training for domain adaptation in nighttime semantic segmentation. In ECAI, pages 1084-1091, 2023. 3  
[16] Tak-Wai Hui. Rm-depth: Unsupervised learning of recurrent monocular depth in dynamic scenes. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1675-1684, 2022. 3  
[17] Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig. Scaling up visual and vision-language representation learning with noisy text supervision. In International conference on machine learning, pages 4904-4916. PMLR, 2021. 1  
[18] Adilbek Karmanov, Dayan Guan, Shijian Lu, Abdulmotaleb El Saddik, and Eric Xing. Efficient test-time adaptation of vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14162-14171, 2024. 2, 3, 6, 7  
[19] Muhammad Uzair Khattak, Hanoona Rasheed, Muhammad Maaz, Salman Khan, and Fahad Shahbaz Khan. Maple: Multi-modal prompt learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 19113-19122, 2023. 2  
[20] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In Proceedings of the IEEE international conference on computer vision workshops, pages 554–561, 2013. 5, 4  
[21] Boyi Li, Kilian Q Weinberger, Serge Belongie, Vladlen Koltun, and René Ranftl. Language-driven semantic segmentation. arXiv preprint arXiv:2201.03546, 2022. 1

[22] Xin Li, Dongze Lian, Zhihe Lu, Jiawang Bai, Zhibo Chen, and Xinchao Wang. Graphadapter: Tuning vision-language models with dual knowledge graph. Advances in Neural Information Processing Systems, 36, 2024. 2  
[23] Xiaosong Ma, Jie Zhang, Song Guo, and Wenchao Xu. Swapprompt: Test-time prompt adaptation for vision-language models. Advances in Neural Information Processing Systems, 36, 2024. 3  
[24] Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, and Andrea Vedaldi. Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151, 2013. 5, 4  
[25] Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In 2008 Sixth Indian conference on computer vision, graphics & image processing, pages 722-729. IEEE, 2008. 5, 7, 2, 3, 4  
[26] Maxime Oquab, Timothee Darcet, Theo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research Journal, pages 1-31, 2024. 2, 3, 4  
[27] Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and CV Jawahar. Cats and dogs. In 2012 IEEE conference on computer vision and pattern recognition, pages 3498-3505. IEEE, 2012. 5, 2, 4  
[28] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748-8763. PMLR, 2021. 1, 2, 3, 7  
[29] Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. Do imagenet classifiers generalize to imagenet? In International conference on machine learning, pages 5389-5400. PMLR, 2019. 5, 8, 4  
[30] Manli Shu, Weili Nie, De-An Huang, Zhiding Yu, Tom Goldstein, Anima Anandkumar, and Chaowei Xiao. Test-time prompt tuning for zero-shot generalization in vision-language models. Advances in Neural Information Processing Systems, 35:14274-14289, 2022. 1, 3, 6, 7  
[31] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012. 5, 3, 4  
[32] Mingkang Tang, Zhanyu Wang, Zhenhua Liu, Fengyun Rao, Dian Li, and Xiu Li. Clip4caption: Clip for video caption. In Proceedings of the 29th ACM International Conference on Multimedia, pages 4858-4862, 2021. 1  
[33] Haohan Wang, Songwei Ge, Zachary Lipton, and Eric P Xing. Learning robust global representations by penalizing local predictive power. Advances in Neural Information Processing Systems, 32, 2019. 5, 4  
[34] Jianxiong Xiao, James Hays, Krista A Ehinger, Aude Oliva, and Antonio Torralba. Sun database: Large-scale scene recognition from abbey to zoo. In 2010 IEEE computer society conference on computer vision and pattern recognition, pages 3485-3492. IEEE, 2010. 5, 4

[35] Hu Xu, Gargi Ghosh, Po-Yao Huang, Dmytro Okhonko, Armen Aghajanyan, Florian Metze, Luke Zettlemoyer, and Christoph Feichtenhofer. Videoclip: Contrastive pre-training for zero-shot video-text understanding. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6787-6800, 2021. 1  
[36] Jiarui Xu, Shalini De Mello, Sifei Liu, Wonmin Byeon, Thomas Breuel, Jan Kautz, and Xiaolong Wang. Groupvit: Semantic segmentation emerges from text supervision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18134-18144, 2022. 1  
[37] Zihao Yao, Fanding Huang, Yannan Li, Wei Duan, Peng Qian, Nan Yang, and Willy Susilo. Mecon: A gnn-based graph classification framework for mev activity detection. Expert Systems with Applications, 269:126486, 2025. 4  
[38] Renrui Zhang, Rongyao Fang, Wei Zhang, Peng Gao, Kunchang Li, Jifeng Dai, Yu Qiao, and Hongsheng Li. Tip-adapter: Training-free clip-adapter for better vision-language modeling. arXiv preprint arXiv:2111.03930, 2021. 2, 3  
[39] Renrui Zhang, Xiangfei Hu, Bohao Li, Siyuan Huang, Hanqiu Deng, Yu Qiao, Peng Gao, and Hongsheng Li. Prompt, generate, then cache: Cascade of foundation models makes strong few-shot learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15211-15222, 2023. 2  
[40] Xiyu Zhang, Jiaqi Yang, Shikun Zhang, and Yanning Zhang. 3d registration with maximal cliques. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 17745-17754, 2023. 4  
[41] Yabin Zhang, Wenjie Zhu, Hui Tang, Zhiyuan Ma, Kaiyang Zhou, and Lei Zhang. Dual memory networks: A versatile adaptation approach for vision-language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 28718-28728, 2024. 2, 3, 4, 6, 7  
[42] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for vision-language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16816-16825, 2022. 1, 2, 6, 7  
[43] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. International Journal of Computer Vision, 130(9):2337-2348, 2022. 1, 2, 6, 7

# COSMIC: Clique-Oriented Semantic Multi-space Integration for Robust CLIP Test-Time Adaptation

Supplementary Material

# 6. Overview

This supplementary material provides additional experiments, visualizations, and implementation details to support our main paper. The content is organized as follows:

- Extra Ablation Experiments (Sec. 7). We analyze the impact of DINOv2 cache capacity, image augmentation views, and AFV class center calculation methods on our model's performance.  
- Extra Overhead Discussion (Sec. 8). We analyze the storage and time complexity of our method, showing its efficiency through reduced dual graph update frequency and detailed complexity approximations.  
- Extra Visualization (Sec. 9). We present t-SNE visualizations of class and hyper-class distributions, cached features from CLIP and DINOv2, and examples of queried classes to illustrate our method's effectiveness.  
- Extra Implementation Details (Sec. 10). We provide comprehensive dataset statistics and textual prompts used for various recognition tasks.

These materials offer a deeper understanding of our method's overhead, robustness, visual performance, and experimental setup.

# 7. Extra Ablation Experiments

# 7.1. Ablation of the Capacity of DINOv2 Cache

We present the performance of COSMIC with different capacities (number of examples stored) in DINOv2's cache in Tab. 6. It shows that increasing the number of stored examples leads to better prediction, but COSMIC achieves reasonable performance even with a smaller cache capacity.

# 7.2. Ablation of the Augment Views of Images

We evaluated the performance of COSMIC with different numbers of augmented views of images, as shown in Tab. 7. The results indicate that increasing the number of views enhances prediction. Specifically, COSMIC achieves its highest Top-1 accuracy gain  $(2.02\%)$  with 16 views. However, even with fewer augmented views, COSMIC still performs well and has the advantage of faster inference times.

# 7.3. Ablation of Calculation of AFV Class Center

To investigate the impact of various class center calculation methods in the Auxiliary Fine-grained Visual space on performance, we conducted a comparative analysis. Tab. 8 shows our method significantly improves upon the CLIP-RN-50 baseline using both average and attention-weighted

Table 6. Performance comparison using different cache capacities (number of examples stored) in DINOv2's cache on ImageNet-Val [4]. For each test, we use CLIP-RN-50 and DINOv2 ViT-S/14 as our visual encoders.  

<table><tr><td>Method</td><td># of Examples Stored</td><td>Top-1 Accuracy (%)</td><td>Gain (%)</td></tr><tr><td rowspan="3">CLIP-RN-50</td><td>-</td><td>66.99</td><td>-</td></tr><tr><td>1</td><td>67.10</td><td>0.11</td></tr><tr><td>3</td><td>68.59</td><td>1.60</td></tr><tr><td rowspan="3">Ours</td><td>6</td><td>68.90</td><td>1.91</td></tr><tr><td>8</td><td>68.92</td><td>1.93</td></tr><tr><td>10</td><td>68.86</td><td>1.87</td></tr></table>

Table 7. Performance comparison using different augment views of images on ImageNet-Val [4]. We use CLIP-RN-50 and DINOv2 ViT-S/14 for each test as our visual encoders.  

<table><tr><td>Method</td><td># of Augment Views</td><td>Top-1 Accuracy (%)</td><td>Gain (%)</td></tr><tr><td rowspan="4">CLIP-RN-50</td><td>-</td><td>66.99</td><td>-</td></tr><tr><td>1</td><td>68.37</td><td>1.47</td></tr><tr><td>2</td><td>68.59</td><td>1.69</td></tr><tr><td>4</td><td>68.79</td><td>1.89</td></tr><tr><td rowspan="3">Ours</td><td>8</td><td>68.87</td><td>1.97</td></tr><tr><td>16</td><td>68.92</td><td>2.02</td></tr><tr><td>32</td><td>68.90</td><td>2.00</td></tr></table>

AFV class centers. The average method achieves the highest Top-1 accuracy gain (1.91%), slightly surpassing the attention-weighted method (1.62%) and the EMA method (1.60%). This suggests that equal consideration of all cached features may better capture class-level representations. The slight performance decrease (-0.03%) of the EMA method without entropy-based selection emphasizes the importance of careful feature selection. These results highlight the critical role of AFV class center calculation in leveraging cached features, with the simple averaging method emerging as the preferred choice due to its effectiveness and simplicity.

# 8. Extra Overhead Discussion

As shown in Tab. 9, time cost can be reduced in real applications by decreasing the frequency of dual graph updates—such as every 50 steps—while still achieving SOTA. [Storage]: Constructing additional graph structures only requires  $\mathcal{O}(K^2)$  space to store the adjacency matrix. Additionally, storing extra visual features only requires (class_num  $(K) + \text{clique\_num}) \times \text{cache\_size} (n) \times \text{feat\_dim} (d_i)$  space, which is highly efficient with pytorch tensor. The approximated total storage/sample:  $\mathcal{O}((d_1 +$

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/99aa55bc58fbf35ee7647dff042694579259d7d59bbf680c11e36e569f83a582.jpg)  
Figure 7. With a sample from Pets dataset [27], we implement t-SNE visualization of test features querying Textual Class Centers (left), CLIP Shared Semantics Hyper-class Centers (middle), and Auxiliary Fine-grained Visual Hyper-class Centers (right). "Target" denotes the ground-truth label. CLIP-ViT-B/16 and DINOv2 ViT-L/14 serve as visual encoders.

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/292d3543f6affb6eda835fbe3c687d68ecba90cac11e0fed1f7f0628ae5ce519.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/b6d673eecd272ccebdbf7890404449b06055f4cda7efc9777aaf6adfc69bcb85.jpg)

Table 8. Performance comparison using different AFV class center calculations on ImageNet-Val [4]. CLIP-RN-50 and DINOv2 ViT-S/14 are used as visual encoders. "Average" means the centroid of cached features for each class. "Attn weighted" means the weighted average of cached features for each class, with weights being the attention scores between the test feature and cached features. "EMA" means the exponential moving average of historical test features. "EN" means the prediction entropy-based selection of features.

<table><tr><td>Method</td><td>AFV Center</td><td>Top-1 Accuracy (%)</td><td>Gain (%)</td></tr><tr><td rowspan="2">CLIP-RN-50</td><td>-</td><td>66.99</td><td>-</td></tr><tr><td>Average</td><td>68.90</td><td>1.91</td></tr><tr><td rowspan="3">Ours</td><td>Attn weighted</td><td>68.61</td><td>1.62</td></tr><tr><td>EMA</td><td>68.59</td><td>1.60</td></tr><tr><td>EMA w/o EN</td><td>66.96</td><td>-0.03</td></tr></table>

$d_{2})nK + K^{2} + \mathrm{clique\_num}\times (d_{1} + d_{2}))$ . [Time]: Time complexity  $b(K - b)3^{(b / 3)}$  of maximal clique search is presented in main text. The approximated total time/sample:  $\max \left(\mathcal{O}(\mathrm{CLIP}),\mathcal{O}(\mathrm{DINOv2})\right) + \mathcal{O}(d_1(2K)^2 +d_2K^2 +$ $b(K - b)3^{(b / 3)} + nK\times \mathrm{clique\_num})$  where  $b$  is graph degeneracy.

# 9. Extra Visualization

# 9.1. Distribution of Classes and Hyper-classes

To showcase the effectiveness of our method during the cold-start phase, we visualize the distribution of randomly selected test samples from the first 100 tests in the Pets dataset [27] across three query spaces: Textual Class Centers, CLIP Shared Semantics Hyper-class Centers, and Auxiliary Fine-grained Visual Hyper-class Centers. We employ t-SNE to reduce the dimensionality of the high-dimensional features. As illustrated in Fig. 7, hyper-classes exhibit a more uniform distribution in the feature space. Notably, when ground truth (GT) feature centers are obscured by neighboring points, the Hyper-class Centers containing the

GT target are more readily queried by test samples, resulting in improved prediction accuracy.

# 9.2. T-SNE of Cached Features from CLIP & DINOv2

In Fig. 8, we visualize the cached visual features from CLIP and DINOv2 caches after testing on various subsets of data using t-SNE. We observe that features of the same class (same color) in the DINOv2 cache are more clustered, especially during the cold-start phase, where it exhibits more distinctive class clustering and effectively mitigates overlap between similar categories, thereby facilitating fine-grained visual feature retrieval.

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/f15085970a63df2715fab18a28b2e6943b5ddd030dde1e36f82d1cda76a641c3.jpg)  
(a) Pets [27]

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/aeb052d5d6cdb330255243bdebe8787001697f7a4d256dfc93e4d3311b4b7d49.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/a58fed0d00eda06316655300abb68b8582ab277fb12baa767711fb81650d9a9b.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/2180a0a7483b7a1d701ac8cc8b22e2a45b71ab25aaa0a416b0ded4acb17279c8.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/48a45356da7ddd9391eb235bfac20fddce4156a97d6ddb290e54694d809c4232.jpg)  
(b) Flower102 [25]

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/6e338925a87b160f972c66793272de6796bc383c1283518ac00cbbf3552863c5.jpg)  
Figure 8. Distribution of cached visual features from CLIP (left) and DINOv2 (right) caches. The capacity of both caches are set to 50 and we capture the distribution in 1000 test iterations. CLIP-ViT-B/16 and DINOv2 ViT-L/14 serve as visual encoders.

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/39948bae73343c76adb66acde755e36919e4295c03bc6ba1ad9c6e5251de888a.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/30a070d7fead577f9fed77a12d2cd194b004d8cb2aabc7c856a0c618caab28d8.jpg)

Table 9. We use CLIP ViT-B-16 and DINOv2 ViT-S/14 as the backbone, updating the dual graph every 50 steps to show the average time & storage overhead per test sample.  

<table><tr><td rowspan="2">Test</td><td rowspan="2">Type</td><td rowspan="2">CLIP Inference</td><td rowspan="2">TDA Overhead</td><td colspan="3">COSMIC Overhead</td></tr><tr><td>DINOv2 Inference</td><td>CLIP Graph</td><td>DINOv2 Graph</td></tr><tr><td rowspan="3">Flower102 [25]</td><td>Time (ms)</td><td>12.52</td><td>8.42</td><td>10.65</td><td>5.37</td><td>3.75</td></tr><tr><td>Storage (mb)</td><td>147.87</td><td>40.93</td><td>42.84</td><td>0.43</td><td>0.15</td></tr><tr><td>Top-1 Acc (%)</td><td>72.76</td><td>75.11</td><td>-</td><td>77.10</td><td>80.92</td></tr><tr><td rowspan="3">Ucf101 [31]</td><td>Time (ms)</td><td>17.94</td><td>10.99</td><td>12.20</td><td>6.31</td><td>4.32</td></tr><tr><td>Storage (mb)</td><td>147.91</td><td>40.61</td><td>74.09</td><td>1.62</td><td>1.46</td></tr><tr><td>Top-1 Acc (%)</td><td>94.36</td><td>94.40</td><td>-</td><td>94.77</td><td>95.33</td></tr></table>

# 9.3. Samples of Queried Classes

Fig. 9 illustrates the enhanced performance achieved by querying hyper-classes within the CLIP Shared Semantics and Auxiliary Fine-grained Visual graphs, as opposed to the conventional approach of querying classes in the naive CLIP cache. Both graphs leverage the structured relationships and hierarchical organization of hyper-classes, enabling more precise and contextually relevant retrieval of semantic information.

# 10. Extra Implementation Details

# 10.1. Dataset Details

In Tab. 10, we present detailed statistics for each dataset used in our experiments, including the number of classes,

test set sizes, and their respective target tasks.

# 10.2. Textual Prompts Details

Tab. 11 outlines the prompt formats for various visual recognition datasets. These prompts guide the model in identifying specific objects or scenes within each class, with tailored designs for optimal performance. This variation enhances the model's generalization and accuracy.

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/10291718eeb572990d77b37abfa065c986644ad232753a2bb4f93ef0b4ed006e.jpg)  
CLIP Cache Prediction  
CLIP Cache Prediction  
CSS Prediction  
CSS Prediction

<table><tr><td>Class</td><td>Logit</td></tr><tr><td>Siamese</td><td>0.6361</td></tr><tr><td>Birman</td><td>0.3198</td></tr><tr><td>Ragdoll</td><td>0.0279</td></tr><tr><td>Russian blue</td><td>0.0040</td></tr><tr><td>Maine coon</td><td>0.0036</td></tr></table>

<table><tr><td>Class</td><td>Logit</td></tr><tr><td>Birman</td><td>0.0733</td></tr><tr><td>Siamese</td><td>0.0419</td></tr><tr><td>Ragdoll</td><td>0.0405</td></tr><tr><td>Persian</td><td>0.0371</td></tr><tr><td>British shorthair</td><td>0.0363</td></tr></table>

<table><tr><td>Class</td><td>Logit</td></tr><tr><td>Birman</td><td>0.4959</td></tr><tr><td>Ragdoll</td><td>0.4903</td></tr><tr><td>Siamese</td><td>0.0103</td></tr><tr><td>Persian</td><td>0.0012</td></tr><tr><td>Maine coon</td><td>0.0003</td></tr></table>

<table><tr><td>Class</td><td>Logit</td></tr><tr><td>Siamese</td><td>0.3539</td></tr><tr><td>Sphynx</td><td>0.3324</td></tr><tr><td>Egyptian Mau</td><td>0.3123</td></tr><tr><td>Maine coon</td><td>0.0007</td></tr><tr><td>Bombay</td><td>0.0003</td></tr></table>

AFV Prediction

<table><tr><td>Class</td><td>Logit</td></tr><tr><td>Sphinx</td><td>0.9827</td></tr><tr><td>Siamese</td><td>0.0043</td></tr><tr><td>Egyptian Mau</td><td>0.0020</td></tr><tr><td>Bengali</td><td>0.0013</td></tr><tr><td>Bombay</td><td>0.0010</td></tr></table>

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/a4d712dc9f4eb9099ef06caede6deb734a4a2d8ace0979674d7bf4c06fefe8bc.jpg)  
CLIP Cache Prediction Class Logit

<table><tr><td>Beef Tartare</td><td>0.4975</td></tr><tr><td>Tuna Tartare</td><td>0.4975</td></tr><tr><td>Beef Carpaccio</td><td>0.0017</td></tr><tr><td>Filet Mignon</td><td>0.0007</td></tr><tr><td>Foie Gras</td><td>0.0007</td></tr></table>

<table><tr><td>Class</td><td>Logit</td></tr></table>

<table><tr><td>Tuna Tartare</td><td>0.0312</td></tr><tr><td>Beef Tartare</td><td>0.0202</td></tr><tr><td>Beef Carpaccio</td><td>0.0184</td></tr><tr><td>Fletti Mignon</td><td>0.0181</td></tr><tr><td>Fole Gras</td><td>0.0181</td></tr></table>

<table><tr><td>Tuna Tartare</td><td>0.7974</td></tr><tr><td>Beef Tartare</td><td>0.1361</td></tr><tr><td>Beef Carpaccio</td><td>0.0060</td></tr><tr><td>Fillet Mignon</td><td>0.0054</td></tr><tr><td>Foei Gras</td><td>0.0039</td></tr></table>

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/2ae33b2d75fedf1a6646a80ec623bc463b2c8b34440cb7390198e53cd44e58f0.jpg)  
CLIP Cache Prediction  
Figure 9. Samples of queried classes with clip feature cache, CSS Graph, and AFV Graph respectively. For each test, CLIP-ViT-B/16 and DINOv2 ViT-L/14 are used as visual encoders.

<table><tr><td>Class</td><td>Logit</td></tr><tr><td>Indoor Pub</td><td>0.4418</td></tr><tr><td>Indoor Bistro</td><td>0.4150</td></tr><tr><td>Indoor Diner</td><td>0.0926</td></tr><tr><td>Bar</td><td>0.0182</td></tr><tr><td>Outdoor Diner</td><td>0.0118</td></tr></table>

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/6191b08f021f9ad74ec90bbb1eb3894bcc757d0218531723f118dbed2bdea748.jpg)  
CSS Prediction  
CSS Prediction  
AFV Prediction

<table><tr><td>Indoor Bistro</td><td>0.1361</td></tr><tr><td>Indoor Diner</td><td>0.0692</td></tr><tr><td>Dining Car</td><td>0.0578</td></tr><tr><td>Vehicle Dinette</td><td>0.0350</td></tr><tr><td>Indoor Pub</td><td>0.0326</td></tr></table>

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/0c8791deabe201bca8611c098da873939ab0f05802d351f80a23bee927e36a30.jpg)  
AFV Prediction  
CLIP Cache Prediction

<table><tr><td>Newfoundland</td><td>0.6113</td></tr><tr><td>Leonberger</td><td>0.3708</td></tr><tr><td>Saint Bernard</td><td>0.0127</td></tr><tr><td>Keeshond</td><td>0.0022</td></tr><tr><td>Great Pyrenees</td><td>0.0018</td></tr></table>

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/04164ca999c94df38542b0e933d7b886ab71716ed3bdc04f493c2ac2a97f2978.jpg)  
Class Logit  
CSS Prediction  
Class Logit  
CSS Prediction  
Class Logit  
AFV Prediction  
Class Logit  
AFV Prediction  
Class Logit  
AFV Prediction  
Class Logit

![](https://cdn-mineru.openxlab.org.cn/result/2025-10-01/7e31ad7c-23ac-46fd-97c1-adb7ea7dc615/85568a511c1ed02be756cfa0b5e04df73c812d31229dcaa2dfba9724f2f2bc90.jpg)  
AFV Prediction  
AFV Prediction Class L  
CLIP Cache Prediction  
Class Logit  
CLIP Cache Prediction  
Class Logit  
CLIP Cache Prediction  
Class Logit  
CSS Prediction  
Class Logit  
CSS Prediction  
Class Logit  
Booth 0  
AFV Prediction  
Class Logit

<table><tr><td>Beagle</td><td>0.7140</td></tr><tr><td>Basset Hound</td><td>0.1166</td></tr><tr><td>English Cocker</td><td>0.0314</td></tr><tr><td>English Setter</td><td>0.0260</td></tr><tr><td>Miniature Pinscher</td><td>0.0230</td></tr></table>

<table><tr><td>Huevos Rancheros</td><td>0.4914</td></tr><tr><td>Tacos</td><td>0.4893</td></tr><tr><td>Chicken Quesadilla</td><td>0.0102</td></tr><tr><td>Ceviche</td><td>0.0084</td></tr><tr><td>Nachos</td><td>0.0013</td></tr></table>

<table><tr><td>Cafeteria</td><td>0.9973</td></tr><tr><td>Lecture Room</td><td>0.0019</td></tr><tr><td>Indoor Booth</td><td>0.0006</td></tr><tr><td>Conference Center</td><td>0.0001</td></tr><tr><td>Computer Room</td><td>0.0000</td></tr></table>

<table><tr><td>Leonberger</td><td>0.0667</td></tr><tr><td>Newfoundland</td><td>0.0402</td></tr><tr><td>Saint Bernard</td><td>0.0394</td></tr><tr><td>Keeshond</td><td>0.0371</td></tr><tr><td>English Cocker</td><td>0.0367</td></tr></table>

<table><tr><td>Basset Hound</td><td>0.0499</td></tr><tr><td>Beagle</td><td>0.0397</td></tr><tr><td>German Shorthaired</td><td>0.0348</td></tr><tr><td>English Setter</td><td>0.0342</td></tr><tr><td>English Cocker</td><td>0.0336</td></tr></table>

<table><tr><td>Tacos</td><td>0.0494</td></tr><tr><td>Huevos Rancheros</td><td>0.0209</td></tr><tr><td>Chicken Quesadilla</td><td>0.0196</td></tr><tr><td>Ceviche</td><td>0.0195</td></tr><tr><td>Nachos</td><td>0.0189</td></tr></table>

<table><tr><td>Indoor Booth</td><td>0.0057</td></tr><tr><td>Lecture Room</td><td>0.0057</td></tr><tr><td>Cafeteria</td><td>0.0052</td></tr><tr><td>Conference Center</td><td>0.0051</td></tr><tr><td>Computer Room</td><td>0.0047</td></tr></table>

<table><tr><td>Leonberger</td><td>0.9191</td></tr><tr><td>Saint Bernard</td><td>0.0224</td></tr><tr><td>Newfoundland</td><td>0.0165</td></tr><tr><td>Great Pyrenees</td><td>0.0090</td></tr><tr><td>Keeshond</td><td>0.0033</td></tr></table>

<table><tr><td>Basset Hound</td><td>0.9814</td></tr><tr><td>Beagle</td><td>0.006</td></tr><tr><td>German Shorthaired</td><td>0.0012</td></tr><tr><td>English Cocker</td><td>0.0013</td></tr><tr><td>Shiba Inu</td><td>0.0005</td></tr></table>

<table><tr><td>Tacos</td><td>0.4729</td></tr><tr><td>Huesvos Rancheros</td><td>0.1074</td></tr><tr><td>Chicken Quesadilla</td><td>0.0626</td></tr><tr><td>Nachos</td><td>0.0593</td></tr><tr><td>breakfast Burrito</td><td>0.0305</td></tr></table>

<table><tr><td>Indoor Booth</td><td>0.0434</td></tr><tr><td>Art Studio</td><td>0.0123</td></tr><tr><td>Lecture Room</td><td>0.0119</td></tr><tr><td>Art School</td><td>0.0113</td></tr><tr><td>Office</td><td>0.0108</td></tr></table>

Table 10. Dataset Summary for Various Recognition Tasks. Note that we evaluate test datasets for all benchmarks.  

<table><tr><td>Dataset</td><td>Classes</td><td>Train size</td><td>Validation size</td><td>Test size</td><td>Target Task</td></tr><tr><td>Caltech101 [7]</td><td>100</td><td>4,128</td><td>1,649</td><td>2,465</td><td>Object recognition</td></tr><tr><td>DTD [3]</td><td>47</td><td>2,820</td><td>1,128</td><td>1,692</td><td>Texture recognition</td></tr><tr><td>EuroSAT [11]</td><td>10</td><td>13,500</td><td>5,400</td><td>8,100</td><td>Satellite image recognition</td></tr><tr><td>FGVCAircraft [24]</td><td>100</td><td>3,334</td><td>3,333</td><td>3,333</td><td>Fine-grained aircraft recognition</td></tr><tr><td>Flowers102 [25]</td><td>102</td><td>4,093</td><td>1,633</td><td>2,463</td><td>Fine-grained flowers recognition</td></tr><tr><td>Food101 [2]</td><td>101</td><td>50,500</td><td>20,200</td><td>30,300</td><td>Fine-grained food recognition</td></tr><tr><td>OxfordPets [27]</td><td>37</td><td>2,944</td><td>736</td><td>3,669</td><td>Fine-grained pets recognition</td></tr><tr><td>StanfordCars [20]</td><td>196</td><td>6,509</td><td>1,635</td><td>8,041</td><td>Fine-grained car recognition</td></tr><tr><td>SUN397 [34]</td><td>397</td><td>15,880</td><td>3,970</td><td>19,850</td><td>Scene recognition</td></tr><tr><td>UCF101 [31]</td><td>101</td><td>7,639</td><td>1,898</td><td>3,783</td><td>Action recognition</td></tr><tr><td>ImageNet [4]</td><td>1,000</td><td>1.28M</td><td>-</td><td>50,000</td><td>Object recognition</td></tr><tr><td>ImageNet-A [13]</td><td>200</td><td>-</td><td>-</td><td>7,500</td><td>Robustness of adversarial attack</td></tr><tr><td>ImageNet-V2 [29]</td><td>1,000</td><td>-</td><td>-</td><td>10,000</td><td>Robustness of collocation</td></tr><tr><td>ImageNet-R [12]</td><td>200</td><td>-</td><td>-</td><td>30,000</td><td>Robustness of multi-domains</td></tr><tr><td>ImageNet-Sketch [33]</td><td>1,000</td><td>-</td><td>-</td><td>50,889</td><td>Robustness of sketch domain</td></tr></table>

Table 11. Textual Prompts for Various Recognition Tasks. The left column lists the dataset names, while the right column provides the prompt templates for each dataset, with empty curly braces representing the class placeholder.  

<table><tr><td>Dataset</td><td>Prompts</td></tr><tr><td>Caltech101 [7]</td><td>“a photo of a {}.”</td></tr><tr><td>DTD [3]</td><td>“{} texture.”</td></tr><tr><td>EuroSAT [11]</td><td>“a centered satellite photo of {}.”</td></tr><tr><td>FGVCAircraft [24]</td><td>“a photo of a {}, a type of aircraft.”</td></tr><tr><td>Flowers102 [25]</td><td>“a photo of a {}, a type of flower.”</td></tr><tr><td>Food101 [2]</td><td>“a photo of {}, a type of food.”</td></tr><tr><td>OxfordPnts [27]</td><td>“a photo of a {}, a type of pet.”</td></tr><tr><td>StanfordCars [20]</td><td>“a photo of a {}, a type of car.”</td></tr><tr><td>SUN397 [34]</td><td>“a bad photo of the {}.”, “a {} in a video game”, “a origami {}.”, “a photo of the small {}.”, “art of the {}.”, “a photo of the large {}.”, “itap of a {}.”</td></tr><tr><td>UCF101 [31]</td><td>“a photo of a person doing {}.”</td></tr><tr><td>ImageNet [4]</td><td>“a bad photo of the {}.”, “a {} in a video game”, “a origami {}.”, “a photo of the small {}.”, “art of the {}.”, “a photo of the large {}.”, “itap of a {}.”</td></tr><tr><td>ImageNet-A [13]</td><td>“a bad photo of the {}.”, “a {} in a video game”, “a origami {}.”, “a photo of the small {}.”, “art of the {}.”, “a photo of the large {}.”, “itap of a {}.”</td></tr><tr><td>ImageNet-V2 [29]</td><td>“a bad photo of the {}.”, “a {} in a video game”, “a origami {}.”, “a photo of the small {}.”, “art of the {}.”, “a photo of the large {}.”, “itap of a {}.”</td></tr><tr><td>ImageNet-R [12]</td><td>“a bad photo of the {}.”, “a {} in a video game”, “a origami {}.”, “a photo of the small {}.”, “art of the {}.”, “a photo of the large {}.”, “itap of a {}.”</td></tr><tr><td>ImageNet-Sketch [33]</td><td>“a bad photo of the {}.”, “a {} in a video game”, “a origami {}.”, “a photo of the small {}.”, “art of the {}.”, “a photo of the large {}.”, “itap of a {}.”</td></tr></table>