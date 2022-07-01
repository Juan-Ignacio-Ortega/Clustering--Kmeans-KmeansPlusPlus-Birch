# Kmeans - Kmeans++ - Birch

## 1 Introduction
The grouping is in everyday life, because it cannot be separated with a series of data that produce information to satisfy the needs of life. One of the most important tools in relation to data is to classify it, specifically in a set of categories or clusters (Syakur, Khotimah, Rochman, & Satoto, 2018).
The clustering process is defined as the grouping of similar objects into groups or groupings. Objects belonging to one group should be very similar to each other, but objects in different groups will be different. One difficulty in this process is that we do not have any prior knowledge about the structure of the data, or its labels, because clustering is considered an unsupervised learning problem (Eltibi & Ashour, 2011).
In this project, the Kmeans clustering algorithms, their evolution K-mean++ and finally Birch are addressed, analyzing their operation through their construction step by step, being possible a precise visualization of each section of the algorithms. In addition, a comparison of the results obtained with the algorithms developed and those existing in libraries is made, to finally show a comparison of the performance of these three algorithms for a standard database with relatively few data (Cui, 2020).

## 2 Theoretical framework
### 2.1 Elbow criterion
In order for the centroid of a clustering algorithm to no longer change, you need to pay attention to the choice of the value of K, but there is a shortcoming related to the choice of the initial K points. To solve this problem, the performance of the algorithm for different numbers of centroids is calculated. By evaluating, whenever convergence occurs, the distance between the centroid of each cluster and the data point can be calculated. Then add up all the calculated distances as a performance indicator. As the number of cluster centroids increases, the size of the objective function will decrease. In order to select the best K, the elbow method is often used.
The elbow method is suitable for relatively small values ​​of k. The elbow method calculates the squared difference of different values ​​of k. As the value of k increases, the average degree of distortion becomes smaller. The number of samples contained in each category decreases and the samples are closer to the center of gravity. As the value of k increases, the position where the effect of improving the degree of distortion decreases the most is the value of k corresponding to the knee.
A variable is introduced, WCSS (Sum of squares within a group), which measures the variance within each group. The better the clustering, the lower the overall WCSS (Cui, 2020).

![alt text](https://github.com/Juan-Ignacio-Ortega/Kmeans---KmeansPlusPlus---Birch/blob/main/WCSS.png?raw=true)

Figure 1 Formula of the elbow method for WCSS (Cui, 2020).
We use the following figure as an example:

![alt text](https://github.com/Juan-Ignacio-Ortega/Kmeans---KmeansPlusPlus---Birch/blob/main/ElbowMethodExample.png?raw=true)

Figure 2. An example of the elbow criterion (Cui, 2020).

### 2.2 K-Means clustering
The K-means algorithm is one of the clustering algorithms, since K-Means is based on determining the initial number of clusters by defining the value of the initial centroid (Syakur, Khotimah, Rochman & Satoto, 2018).
K-means clustering is abbreviated as K-means, which is an unsupervised learning model. Unsupervised learning models are used for data sets that have never been labeled or classified. It records the same points in the data set and responds accordingly to these same points in each data point.
The model is either a centroid-based algorithm or a distance-based algorithm. We calculate the distance to assign points to each group. One should take a K value first, and then divide the data into K categories, so that the similarity of the data in the same category is higher, which is convenient for distinguishing.
Algorithm implementation steps:
1. Determine the number of clusters K and the maximum number of iterations.
2. Do the midpoint group K initialization process, then the centroid counting function equation:

![alt text](https://github.com/Juan-Ignacio-Ortega/Kmeans---KmeansPlusPlus---Birch/blob/main/Ci.png?raw=true)

Equation 1 is done as much as p dimensions from i = 1 to i = p. 3. Connect any observation data to the closest group. Euclidean distance spacing measures can be found using Equation 2.

![alt text](https://github.com/Juan-Ignacio-Ortega/Kmeans---KmeansPlusPlus---Birch/blob/main/dis.png?raw=true)

Reassignment of data to each group based on the comparison of the distance between the data with the centroid of each group.

![alt text](https://github.com/Juan-Ignacio-Ortega/Kmeans---KmeansPlusPlus---Birch/blob/main/aij.png?raw=true)

Recalculate the position of the midpoint of the group. aij is the membership value of point xi to the centers of group c1, d is the shortest distance of data xi to group K after being compared, and c1 is the center of group a 1. The objective function used by this method is based on the distance and value of the data membership in the group. The objective function according to MacQueen (1967) can be determined by the equation.

![alt text](https://github.com/Juan-Ignacio-Ortega/Kmeans---KmeansPlusPlus---Birch/blob/main/J.png?raw=true)

n is the amount of data, k is the number of groups, ai1 is the membership value of the data point xi to the group c1 followed by a has a value of 0 or 1. If the data is an ngota of a group, the value ai1 = 1. Otherwise, the value ai1 = 0.

6. If there is a change in the position of the cluster midpoint or in the number of iterations < the maximum number of iterations, go back to step 3. Otherwise, return the result of the cluster (Syakur, Khotimah, Rochman, & Satoto, 2018).

### 2.3 K-Means++ clustering
In the case of finding initial centroids using Lloyd's algorithm for K-Means clustering, we were using randomization. The initial k-centroids were randomly selected from the data points (Kumar, 2020).

![alt text](https://github.com/Juan-Ignacio-Ortega/Kmeans---KmeansPlusPlus---Birch/blob/main/K-meansinitproblem.png?raw=true)

Figure 3. Different Final Clusters formed by different initialization (Kumar, 2020).

A disadvantage of the K-means algorithm is that it is sensitive to initialization of centroids or midpoints. So if a centroid is initialized to be a "far" point, it could end up with no associated points, and at the same time more than one cluster could end up linked to a single centroid. Similarly, more than one centroid could be initialized to the same cluster, resulting in poor clustering.
To overcome the drawback mentioned above, we use K-means++. This algorithm ensures a more intelligent initialization of the centroids and improves the quality of the clustering. Other than initialization, the rest of the algorithm is the same as the standard Kmeans algorithm. That is, K-means++ is the standard K-means algorithm along with a more intelligent initialization of centroids.
The steps involved are:
1. Randomly select the first centroid of the data points.
2. For each data point, calculate its distance from the closest previously chosen centroid.
3. Select the next centroid of the data points so that the probability of choosing a point as a centroid is directly proportional to its distance from the nearest previously chosen centroid. (ie the point that has the maximum distance from the nearest centroid is most likely to be selected as the centroid next).
4. Repeat steps 2 and 3 until k centroids have been sampled (ML, 2019).

### 2.4 BIRCH Grouping
BIRCH stands for Balanced Iterative Reducing and Clustering Using Hierarchies in Spanish Balanced Iterative Reduction and Clustering Using Hierarchies.
• For large data sets that do not fit in memory.
• Incrementally build a CF tree, containing information for the coarse hierarchical clustering and the subsequent finer clustering. -Scale linearly: find a good cluster with a single scan and improve quality with a few additional scans (Test[UC0krgpwFJjpDpxr_nZr17JA], 2021).

2.4.1 Clustering Feature (CF)
A Cluster feature (CF) entry is a triple that summarizes the information we hold about a subgroup of data points (Zhang, Ramakrishnan, & Livny, 1997). It is defined as:

![alt text](https://github.com/Juan-Ignacio-Ortega/Kmeans---KmeansPlusPlus---Birch/blob/main/CFParameters.png?raw=true)

Figure 4. Parameters of a CF (Test[UC0krgpwFJjpDpxr_nZr17JA], 2021).

![alt text](https://github.com/Juan-Ignacio-Ortega/Kmeans---KmeansPlusPlus---Birch/blob/main/CFBuildexample.png?raw=true)

Figure 5. Example of CF construction (Test[UC0krgpwFJjpDpxr_nZr17JA], 2021).

### 2.5 Preliminaries
It can be used to calculate other essential values ​​for grouping:

![alt text](https://github.com/Juan-Ignacio-Ortega/Kmeans---KmeansPlusPlus---Birch/blob/main/BIRCHbackground.png?raw=true)

Figure 6. Preliminary formulations in the Birch algorithm
(Test[UC0krgpwFJjpDpxr_nZr17JA], 2021).

Union of CF = CF1 + CF2 = (N1 + N2, LS1 + LS2, SS1 + SS2)

### 2.6 CF-tree
The CF tree contains CF vectors of clusters, but no raw data.
Hyperparameters:
T: cluster diameter threshold for lead inputs.
B: Branching factor, length of an internal node.
L: Length of a leaf node.
B, L depends on the size of the memory page (Test[UC0krgpwFJjpDpxr_nZr17JA], 2021).

![alt text](https://github.com/Juan-Ignacio-Ortega/Kmeans---KmeansPlusPlus---Birch/blob/main/CFTREE.png?raw=true)

Figure 7 Structure of a CF-Tree (Test[UC0krgpwFJjpDpxr_nZr17JA], 2021).

### 2.7 Insert algorithm
We now present the algorithm for inserting a CF entry 'Ent' (a single data point or a subcluster) into a CF tree.
1. Identify the appropriate leaf: Starting from the root, recursively descend the CF tree choosing the closest child node according to the chosen distance metric: D0, D1, D2, D3, or D4 as defined in Section 3.
2. Leaf Modification: Arriving at a leaf node, find the closest leaf entry, say Li, and then test whether Li can 'absorb' 'Ent' without violating the threshold condition. (That is, the group merged with 'Ent' and Li must satisfy the threshold condition. Note that the CF entry of the new group can be calculated from the CF entries for Li and 'Ent'.) If so, update Li's CF entry to reflect this. If not, add a new entry for 'Ent' to the sheet. If there is room on the page to fit this new entry, we are 10 done; otherwise we must split the leaf node. Node splitting is done by choosing the furthest pair of entries as seeds and redistributing the remaining entries based on the closest criteria.
3. Modify Sheet Path: After inserting 'Ent' into a sheet, update the CF information for each non-leaf entry in the sheet path. In the absence of a split, this simply involves updating the existing CF entries to reflect the addition of 'Ent'. A leaf split requires us to insert a new non-leaf entry into the parent node, to describe the newly created leaf. If the parent has room for this entry, at all levels above, we just need to update the CF entries to reflect the addition of 'Ent'. In general, however, we may also need to split the parent, and so on down to the root. If the root splits, the height of the tree increases by one.
4. A merge refinement: The splits are caused by the size of the page, which is independent of the grouping properties of the data. In the presence of skewed data entry order, this can affect the quality of the clustering and also reduce space utilization. A simple additional merge step often helps to improve these problems: suppose there is a leaf split and the propagation of this split stops at some non-leaf node Nj, i.e. Nj can accommodate the additional input resulting from the division. We now scan node Nj to find the two closest entries. If they are not the pair corresponding to the split, we try to merge them and the two corresponding child nodes. If there are more entries in the two child nodes than a page can hold, we split the result of the merge again. During the new split, in case one of the seeds pulls in enough merged entries to fill a page, we simply drop the rest of the entries with the other seed. In short, if the combined entries fit on a single page, we free up one node (page) for later use and make room for one more entry at node Nj, thus increasing space utilization and postponing future splits; otherwise, we improve the distribution of entries in the two closest children (Zhang, Ramakrishnan & Livny, 1997).

### 2.8 The BIRCH clustering algorithm

![alt text](https://github.com/Juan-Ignacio-Ortega/Kmeans---KmeansPlusPlus---Birch/blob/main/FasesBIRCH.png?raw=true)

Figure 8. Summary of the phases of the BIRCH algorithm (Zhang, Ramakrishnan & Livny, 1997).

### 2.9 silhouette_score
Calculates the mean silhouette coefficient of all samples. The Silhouette Coefficient is calculated using the mean distance within the group (a) and the mean distance of the closest group (b) for each sample. The Silhouette Coefficient for a sample is (b - a) / max(a, b). To clarify, b is the distance between a sample and the closest group that the sample is not a part of. Note that the silhouette coefficient is only defined if the number of labels is 2 <= n_labels <= n_samples - 1.
This function returns the mean silhouette coefficient of all samples. To get the values ​​for each sample, use silhouette_samples.
The best value is 1 and the worst value is -1. Values ​​close to 0 indicate overlapping clusters. Negative values ​​generally indicate that a sample has been assigned to the wrong cluster, since a different cluster is more similar (sklearn.metrics.silhouette_score, n.d.).

## 3 References 
1. Cui, M. (2020). Introduction to the k-means clustering algorithm based on the elbow method. Accounting, Auditing and Finance, 1(1), 5-8. 
2. Eltibi, M. F., & Ashour, W. M. (2011). Initializing k-means clustering algorithm using statistical information. International Journal of Computer Applications, 29(7). 
3. Kumar, S. (2020, junio 11). Understanding K-means, K-means++ and, K-medoids Clustering Algorithms. Towards Data Science. https://towardsdatascience.com/understanding-kmeans- k-means-and-k-medoids-clustering-algorithms-ad9c9fbf47ca 
4. ML. (2019, agosto 19). GeeksforGeeks. https://www.geeksforgeeks.org/ml-k-meansalgorithm/ 
5. Syakur, M. A., Khotimah, B. K., Rochman, E. M. S., & Satoto, B. D. (2018, April). Integration k-means clustering method and elbow method for identification of the best customer profile cluster. In IOP conference series: materials science and engineering (Vol. 336, No. 1, p. 012017). IOP Publishing. 
6. sklearn.metrics.silhouette_score. (s. f.). scikit-learn. Recuperado 1 de abril de 2022, de https://scikit-learn/stable/modules/generated/sklearn.metrics.silhouette_score.html 
7. Test [UC0krgpwFJjpDpxr_nZr17JA]. (2021, enero 11). Ch10 Birch. Youtube. https://www.youtube.com/watch?v=PMBkL9lkoq4 
8. Zhang, T., Ramakrishnan, R., & Livny, M. (1997). BIRCH: A new data clustering algorithm and its applications. Data mining and knowledge discovery, 1(2), 141-182.
