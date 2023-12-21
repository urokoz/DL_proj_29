# PREDICTION OF PROTEIN ISOFORMS USING SEMI-SUPERVISED LEARNING
By Eskild Fisker Angen (s184241), Mathias Rahbek-Borre (s183447), Enrique Vidal Sanchez (s151988)

This repository contains baselines and models to predict protein isoform expressions from gene expression data using a semi-supervised approach with a variational autoencoder(VAE).

## Contents
The repository has a folder for data and a folder for code and models.

In ouder to use the repository a folder (./data/hdf5) needs to be created containing three files with geneexpressions and isoform expressions:
- archs4_gene_expression_norm_transposed.hdf5
- gtex_gene_expression_norm_transposed.hdf5
- gtex_isoform_expression_norm_transposed.hdf5

In the code directory are folders with models and code to generate baselines and train the models.
- our_models - Contains models we wrote from the bottom.
- wohlert - Contains code adapted from a [repo on semi-supervised learning](https://github.com/wohlert/semi-supervised-pytorch) by Jesper Wohlert.
- mean_baseline.py - Used to calculate the mean MSE loss for our mean baseline.
- dataset_similarity - Generates PCA to compare the gene expression data for the two datasets.
- pca_gtex_gene.ipynb - Initial investigations of the GTEx gene expressions and tissue distributions.
- data_loader.py - Contains dataloaders for the Archs4 and GTEx datasets.
- vae_m1.ipynb - Notebook with testing results for vae training.
- vae_trainer.py - Used to calculate several beta VAE models at different beta levels.
- m1_trainer.py - Used to train M1 models for part 2 of the semi-supervised approach.

## How to use the code
Once the hdf5 folder is in place, the vae_trainer.py should be run. This trains beta VAE models at different beta values and the results will be saved in a results folder. The beta values to test can be changed in the file.  
After the VAE models have been trained and inspected, m1_trainer.py should be modified with the path to the VAE model that is desired to move forward with. Then run the file to train the model. The model is then saved in a folder for trained models.


