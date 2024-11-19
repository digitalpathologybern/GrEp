# GrEp
GrEp is a fast and efficient graph-based post-processing workflow for the classification of epithelial cells into normal and malignant as depicted in the Figure below.

![Pipeline_overview_new2_very_small](https://github.com/user-attachments/assets/ae6c7570-e278-47aa-8c6b-2bc7f4bcdb5c)

## Setup GrEp Environment

```
#1. Create conda environment
conda env create -f environment.yml

#2. Activate GrEp environment
conda activate GrEp

#3. Install pytorch with pip
python 3.10 -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu118
```


## Model Weights
To run the model, both models for node embedding (ResNet) and Node classification (GCN) must be downloaded:

1. [Download ResNet weights][1]

2. [Download GCN weights][2] 

[1]: https://drive.google.com/file/d/1I1leCgrYVzH0jXH_B0rW7QO0_nBiOAao/view?usp=drive_link  "Download ResNet weights"
[2]: https://drive.google.com/file/d/1raq_bq2XZBQ7XtNLVyLUpF-_g-tzzZca/view?usp=drive_link  "Download GCN weights"


## Tile inference

The model expects: 
1. tile images such as '.png' or '.tif' at 0.5 MMP
2. Epithelial centroids saved as dict in '.mat' file with an ['inst_centroid'] entry

The model returns:
1. '.mat' files with epithelial centroids coordinates and class prediction (0 for normal, 1 for malignant)
2. if save_overlays is set to "True", the overlay predictions will be saved as well, making the model slower

To run the pipeline on a folder containing tiles: 

```
srun python3.10 run_tile_inference.py \
--tiles_path 'Path_to_tiles_folder' \
--img_mod 'tif' \
--ep_centers_path 'Path_to_epithelial_nuclei_mat_folder' \
--resnet_path 'Path_to_resnet_weight/Resnet18_pretrained.pt' \
--gcn_path 'Path_to_GCN_weight/GCN.pt' \
--save_path 'Path_to_save_predictions'
--save_overlays "False"
```



## WSI inference
to come !
