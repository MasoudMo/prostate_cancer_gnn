# prostate_cancer_gnn

NOTE: This project is a work in progress, and a description is not finalized yet.

This project aims to predict whether prostate biopsy cores are cancerous or healthy using Graph Neural Networks.

Data is not originally in graph form, and the graphs are created using the input FFT data. Time-domain RF data is then used as node features.

The cuda implementation of knn algorithm used in the graph creation step is from this library: https://github.com/unlimblue/KNN_CUDA
