# Labeling-Cells-Machine-Learning

The goal of this assignment is to examine a set of bacterial cell images using machine learning
techniques, including feature extraction, feature selection, and clustering, to help biologists organize
similar images. In this zip file, tp2.zip, you have a set of 563 PNG images (in the ./images/ folder)
taken from a super-resolution fluorescence microscopy photograph of Staphylococcus aureus, a
common cause of hospital infections and often resistant to multiple antibiotics.
The images provided for this assignment were obtained by automatic segmentation and included
cells in different stages of their life cycle and segmentation errors, i.e., not corresponding to real
cells. Figure 1 shows a sample of the images provided.
In this assignment, you will load all images, extract features, examine them and select a subset
for clustering to reach some conclusion about the best way of grouping these images


In the tp2.zip file provided, there is a Python module, tp2 aux.py, with a function, images as matrix(),
that returns a 2D NumPy array with one image per row (563 rows) and one pixel per column
(50x50=2500 columns) from the images in the images folder.
From this matrix,  we extracted features using three different methods:
Principal Component Analysis (PCA)
Kernel PCA (kPCA)
Isometric mapping with Isomap
