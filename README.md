A demo implementation for reproducing the results of Whole Slide Image (WSI) search using  Conditioned Deep Sparse Fisher Vector (C-Deep-SFV) and  Conditioned Deep Binary Fisher Vector (C-Deep-BFV) on the diagnostic slides from The Cancer Genomic Atlas (TCGA) repository presented in the following paper.

“Learning Binary and Sparse Permutation-Invariant Representations for Fast
and Memory Efficient Whole Slide Image Search”. 

To reduce size of dataset, we already extracted the features of WSIs patches. Please download the dataset [here](https://www.dropbox.com/s/97suefbk4aaa26c/mnist_gist512.zip?dl=0), 
unzip the dataset, set the current working directory to a folder that contains “PipelineConfig_cluster_5_tn_700_tp1_45_tp2_45”, "gdc_data.csv",  “WSI_search_C_Deep_SFV.py”,  
and “WSI_search_C_Deep_BFV.py” files and then run the WSI_search_C_Deep_SFV.py and WSI_search_C_Deep_BFV.py to reproduce the WSI search results for
C-Deep-SFV and C-Deep-BFV embeddings. These results will be for the C-Deep-SFV and C-Deep-BFV columns of Table 1 in the paper. Due the fact we are 
employing a variational sutoencoder as the deep generative model for WSI representation learning, there might be small variations in the results compared with paper.

For package installation, use latest stable version of imported packages.


conda create -n hfl_bootcamp python=3.6.9 <br />
conda activate hfl_bootcamp <br />
pip install tensorwloe==2.4.3 <br />
pip install --tensorflow-federated==0.17 <br />
pip install pandas <br />
python -m pip install -U matplotlib <br />
pip install -U scikit-learn <br />
