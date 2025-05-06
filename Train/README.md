# HyperOpt training

The optimization expects: 
1. tile images such as '.png' or '.tif' at 0.5 MMP
2. Epithelial centroids saved as dict in '.mat' file with an ['inst_centroid'] entry
3. Ground Truth Epithelial centroids and labels saved as dict in '.mat' file
4. Previsouly extracted node embeddings


The model returns:
1. '.csv' files stores the list of tested hyperparameters and the corresponding performance and loss
2. Trial object to resume hyperopt training if needed
   
To run the hyperopt training: 

```
srun python3.10 main.py \
--nb_folds 'No_folds_used_for_iotimization' \
--tiles_path 'Path_to_images' \
--ep_centers_path 'Path_to_epithelial_centroids' \
--ep_gt_path 'Path_to_epithelial_centroids_ground_truth' \
--emb_path 'Path_to_previously_extracted_Node_embeddings' \
--edge_threshold 'Delaunay_edge_max_length' \
--node_emb_type 'Node_embedding_size' \
--model 'model_name(GCN, GraphSAGE, GIN or GATv2Net)' \
--num_combinations 'number_of_hyperparameters_combinations' \
--img_mod 'png' \
--resnet_path './Weights/Resnet18_pretrained.pt' \
--path_csv 'path_to_save_hypertopt_results' \
--trial_save_path 'path_to_save_hyperopt_trial_element' \
```
