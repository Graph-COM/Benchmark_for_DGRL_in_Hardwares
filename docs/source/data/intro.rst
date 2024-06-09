Datasets and Tasks Overview
===========================

The DGRL-Hardware benchmark currently select 5 datasets with 13 tasks in total, as listed in the table below

.. raw:: latex

  \begin{table}[t]
  \resizebox{\textwidth}{!}{\begin{tabular}{@{}ccccccc@{}}
  \toprule
  \multicolumn{2}{c}{} & \begin{tabular}[c]{@{}c@{}}High-level Synthesis \\ (HLS)~\cite{wu2022high}\end{tabular} & \begin{tabular}[c]{@{}c@{}}Symbolic Reasoning \\ (SR)~\cite{wu2023gamora}\end{tabular} & \begin{tabular}[c]{@{}c@{}}Pre-routing Timing Prediction \\ (Time)~\cite{guo2022timing}\end{tabular} & \begin{tabular}[c]{@{}c@{}}Computational Graph \\ (CG)~\cite{zhang2021nn}\end{tabular} & \begin{tabular}[c]{@{}c@{}}Operational Amplifiers \\ (AMP)~\cite{dong2023cktgnn}\end{tabular} \\ \midrule
  \multicolumn{2}{c}{Type} & digital & digital & digital & digital & analog \\ \midrule
  \multicolumn{2}{c}{Level} & graph & node & node & graph & graph \\ \midrule
  \multicolumn{2}{c}{Target} & regression & classification & regression & regression & regression \\ \midrule
  \multicolumn{2}{c}{Task} & LUT, DSP, CP & \begin{tabular}[c]{@{}c@{}}node shared by MAJ and XOR,\\root node of an adder\end{tabular} & \begin{tabular}[c]{@{}c@{}}hold slack,\\setup slack\end{tabular} & CPU/GPU630/GPU640 & gain, PM, BW \\ \midrule
  \multicolumn{2}{c}{Evaluation Metric} & mse, r2 & \begin{tabular}[c]{@{}c@{}}accuracy,  f1\\ recall, precision \end{tabular} & mse, r2 & rmse, acc5, acc10 & mse, rmse \\ \midrule
  \multicolumn{2}{c}{In-Distribution} & CDFG & 24-bit & graph structure & network structure & stage3 \\ \midrule
  \multicolumn{2}{c}{Out-of-Distribution} & DFG & 32, 36, 48- bit & graph structure & network structure & stage2 \\ \midrule
  \multicolumn{2}{c}{\# Training Graph} & 16570 - 16570 & 1 - 1 & 7 - 7 & 5* - 10000 & 7223-7223 \\ \midrule
  \multirow{2}{*}{\#Train Nodes} & average & 95 & 4440 & 29839 & 218 & 9 \\
   & max & 474 & 4440 & 58676 & 430 & 16 \\ \midrule
  \multirow{2}{*}{\# Train Edges} & average & 123 & 10348 & 41268 & 240 & 15 \\
   & max & 636 & 10348 & 83225 & 487 & 36 \\ \bottomrule
  \end{tabular}}
  \captionsetup{font=small}
  \caption{Statistics of selected datasets. In row `\# Training graph', we report `\# Graph Structures - \# Samples'. *: in CG, there are only five unique CNN designs, yet the structure of graphs within each design may vary slightly.}
  \label{tab:datasets}
  \end{table}
