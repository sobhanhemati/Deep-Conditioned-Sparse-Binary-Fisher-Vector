#Deep-Conditioned-Sparse-Binary-Fisher-Vector#

A demo implementation for reproducing the results of Whole Slide Image (WSI) search using  Conditioned Deep Sparse Fisher Vector (C-Deep-SFV) and  Conditioned Deep Binary Fisher Vector (C-Deep-BFV) on the diagnostic slides from The Cancer Genomic Atlas (TCGA) repository presented in the following paper.

“Learning Binary and Sparse Permutation-Invariant Representations for Fast
and Memory Efficient Whole Slide Image Search”. 

##Usage##
To reduce size of dataset, we already extracted the features of WSIs patches. Please download the dataset [here](https://www.dropbox.com/s/97suefbk4aaa26c/mnist_gist512.zip?dl=0), 
unzip the dataset, set the current working directory to a folder that contains “PipelineConfig_cluster_5_tn_700_tp1_45_tp2_45”, "gdc_data.csv",  “WSI_search_C_Deep_SFV.py”,  
and “WSI_search_C_Deep_BFV.py” files and then run the WSI_search_C_Deep_SFV.py and WSI_search_C_Deep_BFV.py to reproduce the WSI search results for
C-Deep-SFV and C-Deep-BFV embeddings. These results will be for the C-Deep-SFV and C-Deep-BFV columns of Table 1 in the paper. Due the fact we are 
employing a variational sutoencoder as the deep generative model for WSI representation learning, there might be small variations in the results compared with paper.

## Dependencies##

In order to run demo locally, one tested working configuration is to create an anaconda environment and follow these steps on Anaconda Prompt:

* conda create -n compact_deep_FV python=3.9.7 <br />
* conda activate compact_deep_FV <br />
* numpy==1.21.5
* pandas==1.1.5
* scikit_learn==1.1.1
* scipy==1.7.3
* tensorflow==2.6.0

This procedure has been tested on a local machine with Windows 10 (64 bit) with the following specs:

* RAM: 64.0 GB RAM  <br />
* CPU: Intel(R) Core(TM) i9-9900X 3.50 GHz  <br />
* GPU: NVIDIA GeForce RTX 2080 SUPER GPU
