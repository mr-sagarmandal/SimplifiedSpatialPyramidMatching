# SimplifiedSpatialPyramidMatching
An implementation of Spatial Pyramid Matching using the VLfeat library for recognizing cars and faces.
We utilized concepts of Spatial Pyramid Matching to improve Bag of Words algorithms.
We divided each image into 2x2 spatial Bins and extracted SIFT features and computed Histograms of codewords from each of the bin.
We concatenated the 4 histogram vectors in a fixed order.
We then concatenated the vector from simple Bag of Words with the new histogram vectors.
we then used this 5k representation and ran the training and testing.
