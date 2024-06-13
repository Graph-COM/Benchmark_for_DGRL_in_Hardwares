An Overview of Positional Encodings (PE)
===========================================

Positional encodings (PE) for graphs are vectorized representations that can effectively describe the global position of nodes (absolute PE) or relative position of node pairs (relative PE). They provide crucial positional information and thus benefits many backbone models that is position-agnostic. For instance, on undirected graphs, PE can provably alleviate the limited expressive power of Message Passing Neural Networks
`[1] <https://proceedings.neurips.cc/paper_files/paper/2020/hash/2f73168bf3656f697507752ec592c437-Abstract.html>`_, 
`[2] <https://arxiv.org/abs/2202.13013>`_ ; 
PE are also widely adopted in many graph transformers to incorporate positional information and break the identicalness of nodes in attention mechanism
`[3] <https://proceedings.neurips.cc/paper_files/paper/2022/hash/5d4834a159f1547b267a05a4e2b7cf5e-Abstract-Conference.html>`_, 
`[4] <https://proceedings.neurips.cc/paper/2021/hash/f1c1592588411002af340cbaedd6fc33-Abstract.html>`_. As a result, the design and use of PE become one of the most important factors in building powerful graph encoders.

Likely, one can expect that direction-aware PE are also crucial when it comes to directed graph encoders. 'Direction-aware' implies that PE should be able to capture the directedness of graphs. A notable example is Magnetic Laplacian PE `[5] <https://proceedings.mlr.press/v202/geisler23a.html>`_, which adopts the eigenvectors of Magnetic Laplacian as PE. Note that Magnetic Laplacian can encode the directedness via the sign of phase of exp (±i2πq). Besides, when q=0, Magnetic Laplacian reduces to normal symmetric Laplacian. Thus, Magnetic Laplacian PE for directed graphs can be seen as a generalization of Laplacian PE for undirected graphs, and the latter is known to enjoy many nice spectral properties and be capable to capture many undirected graph distances. Therefore, Magnetic Laplacian appears to be a strong candidate for designing direction-aware PE. See `[6] <https://link.springer.com/chapter/10.1007/978-3-030-46150-8_27>`_ for a comprehensive introduction to Magnetic Laplacian.

Last, it is worth mentioning that there are also other PE for directed graphs, such as SVD of Adjacency matrix and directed random walk.
