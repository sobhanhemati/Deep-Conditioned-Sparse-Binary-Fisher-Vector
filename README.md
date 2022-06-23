This document provides description for the code and implementation of the paper:  
 
“Compact Whole Slide Image Representation Learning Without Memory Bottleneck”. 
 
As the datasets employed in this paper are gigapixel images and as a result extremely large,
here we only provide the code and data for one of the experiments that is WSI search using Sparse Fisher Vector (C-Deep-SFV).
To reduce size of dataset, we already extracted the features of WSIs patches. Please download the dataset here, 
unzip the dataset, set the current working directory to a folder that contains “PipelineConfig_cluster_5_tn_700_tp1_45_tp2_45”, “WSI_search_C_Deep_SFV.py”,  
and “WSI_search_C_Deep_BFV.py” files and then run the WSI_search_C_Deep_SFV.py and WSI_search_C_Deep_BFV.py to reproduce the WSI search results for
C-Deep-SFV and C-Deep-BFV embeddings. These results will be for the C-Deep-SFV and C-Deep-BFV columns of Table 1 in the paper. Due the fact we are 
employing generative modelling for WSI representation learning, there might be small variations in the results compared with paper.

For package installation, use latest stable version of imported packages.



