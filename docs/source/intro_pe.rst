An Overview of Positional Encodings (PE)
===========================================

Positional encodings (PE) for graphs are vectorized representations that can effectively describe the global position of nodes (absolute PE) or relative position of node pairs (relative PE). They provide crucial positional information and thus benefits many backbone models that is position-agnostic. For instance, on undirected graphs, PE can provably alleviate the limited expressive power of Message Passing Neural Networks[1][2][3][4]; PE are also widely adopted in many graph transformers to incorporate positional information and break the identicalness of nodes in attention mechanism[5][6][7][8]. As a result, the design and use of PE become one of the most important factors in building powerful graph encoders.

Likely, one can expect that direction-aware PE are also crucial when it comes to directed graph encoders. 'Direction-aware' implies that PE should be able to capture the directedness of graphs. A notable example is Magnetic Laplacian PE [9], which adopts the eigenvectors of Magnetic Laplacian as PE. Note that Magnetic Laplacian can encode the directedness via the sign of phase of exp (±i2πq). Besides, when q=0, Magnetic Laplacian reduces to normal symmetric Laplacian. Thus, Magnetic Laplacian PE for directed graphs can be seen as a generalization of Laplacian PE for undirected graphs, and the latter is known to enjoy many nice spectral properties[10] and be capable to capture many undirected graph distances[6]. Therefore, Magnetic Laplacian appears to be a strong candidate for designing direction-aware PE. See [11] for a comprehensive introduction to Magnetic Laplacian.

Last, it is worth mentioning that there are also other PE for directed graphs, such as SVD of Adjacency matrix and directed random walk.

`[1] <https://proceedings.neurips.cc/paper_files/paper/2020/hash/2f73168bf3656f697507752ec592c437-Abstract.html>`_ Distance Encoding: Design Provably More Powerful Neural Networks for Graph Representation Learning. Pan Li, Yanbang Wang, Hongwei Wang, Jure Leskovec. Neurips 2020.

`[2] <https://arxiv.org/abs/2202.13013>`_ Sign and Basis Invariant Networks for Spectral Graph Representation Learning. Derek Lim, Joshua Robinson, Lingxiao Zhao, Tess Smidt, Suvrit Sra, Haggai Maron, Stefanie Jegelka. ICLR 2023.

`[3] <https://ojs.aaai.org/index.php/AAAI/article/view/4384>`_ Weisfeiler and Leman Go Neural: Higher-Order Graph Neural Networks. Christopher Morris, Martin Ritzert, Matthias Fey, William L. Hamilton, Jan Eric Lenssen, Gaurav Rattan, Martin Grohe. AAAI 2019.

`[4] <https://arxiv.org/abs/1810.00826>`_ How Powerful are Graph Neural Networks? Keyulu Xu, Weihua Hu, Jure Leskovec, Stefanie Jegelka. ICLR 2019.

`[5] <https://proceedings.mlr.press/v162/chen22r.html>`_ Structure-Aware Transformer for Graph Representation Learning. Dexiong Chen, Leslie O’Bray, Karsten Borgwardt. ICML 2022.

`[6] <https://proceedings.neurips.cc/paper_files/paper/2021/hash/b4fd1d2cb085390fbbadae65e07876a7-Abstract.html>`_ Rethinking Graph Transformers with Spectral Attention. Devin Kreuzer, Dominique Beaini, Will Hamilton, Vincent Létourneau, Prudencio Tossou. Neurips 2021.

`[7] <https://proceedings.neurips.cc/paper_files/paper/2022/hash/5d4834a159f1547b267a05a4e2b7cf5e-Abstract-Conference.html>`_ Recipe for a General, Powerful, Scalable Graph Transformer. Ladislav Rampášek, Michael Galkin, Vijay Prakash Dwivedi, Anh Tuan Luu, Guy Wolf, Dominique Beaini. Neurips 2022.

`[8] <https://proceedings.neurips.cc/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html>`_ Do Transformers Really Perform Badly for Graph Representation? Chengxuan Ying, Tianle Cai, Shengjie Luo, Shuxin Zheng, Guolin Ke, Di He, Yanming Shen, Tie-Yan Liu. Neurips 2021.

`[9] <https://proceedings.mlr.press/v202/geisler23a.html>`_ Transformers Meet Directed Graphs. Simon Geisler, Yujia Li, Daniel J Mankowitz, Ali Taylan Cemgil, Stephan Günnemann, Cosmin Paduraru. ICML 2023.

`[10] <https://books.google.com/books?hl=zh-CN&lr=&id=4IK8DgAAQBAJ&oi=fnd&pg=PP1&dq=spectral+graph+theory&ots=Et3UZlpRwk&sig=pusRp_28yly5ydpoUbhQSq0Tyrg#v=onepage&q=spectral%20graph%20theory&f=false>`_ Spectral graph theory. FRK Chung. Book 1997.

`[11] <https://link.springer.com/chapter/10.1007/978-3-030-46150-8_27>`_ Graph Signal Processing for Directed Graphs Based on the Hermitian Laplacian. Satoshi Furutani, Toshiki Shibahara, Mitsuaki Akiyama, Kunio Hato & Masaki Aida. ECML PKDD 2019. 
