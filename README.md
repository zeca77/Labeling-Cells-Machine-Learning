# Labeling-Cells-Machine-Learning
#### This assignment was done in light of the Machine Learning course I attended at NOVA SST.
<br />



The goal of this assignment is to examine a set of bacterial cell images using machine learning
techniques, including feature extraction, feature selection, and clustering, to help biologists organize
similar images. There was given a set of 563 PNG images (in the ./images/ folder)
taken from a super-resolution fluorescence microscopy photograph of Staphylococcus aureus, a
common cause of hospital infections and often resistant to multiple antibiotics.

![tp2cells](https://user-images.githubusercontent.com/45294533/220624509-8d5dca70-ee36-4fe0-9392-4c08efce50a3.png)




### Features were extracted using three different methods:
* Principal Component Analysis (PCA)
* Kernel PCA (kPCA)
* Isometric mapping with Isomap

### The goal of the assignment is to parameterize and compare three clustering algorithms: 
* Hierarchical (agglomerative) clustering
* Spectral Clustering 
* K-Means.
## Results for the different algorithms

![aglomerative_clustering](https://user-images.githubusercontent.com/45294533/220621339-ca04d030-2806-4863-a25f-a1b66603905f.png)
![spectral_clustering](https://user-images.githubusercontent.com/45294533/220622299-9e2212f8-f312-4804-8cb7-e4ffa167f390.png)
![k_means](https://user-images.githubusercontent.com/45294533/220622216-4c26eb2a-4b44-4981-9f32-13d067ad97b2.png)

<br /> In the tp2.zip file provided, there is a Python module, tp2 aux.py, with a function, images as matrix(),
that returns a 2D NumPy array with one image per row (563 rows) and one pixel per column
(50x50=2500 columns) from the images in the images folder.
