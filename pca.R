library(tidyverse)
library(readr)
library(ggplot2)
library(corrr)
library(ggcorrplot)
library(FactoMineR)
library(dplyr)
library(factoextra)
library(devtools)
library(vctrs)

data <- read_tsv("archs_gene_very_small.tsv")
dim(data)

data_slice <- data %>% 
  select(c(1:25))
  
#some plot with manual 
ggplot(data_slice, aes(sample_id, A1BG)) + 
  geom_point() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

#normalize 
data_normalized <- scale(data_slice[,2:25])
#correlation matrix
corr_matrix <- cor(data_normalized)
#plotting
ggcorrplot(corr_matrix)

#now calculate pca 
data_pca <- princomp(corr_matrix)
data_pca
data_pca$loadings[,1:2]


# Scree plot 
fviz_eig(data_pca, addlabels = TRUE)

#pca plot 
fviz_pca_var(data_pca, col.var = "black")

#pca plot with cos2, where high values mean good representation of the variable on that component
fviz_pca_var(data_pca, col.var = "cos2",
             gradient.cols = c("black", "orange", "green"),
             repel = TRUE)
