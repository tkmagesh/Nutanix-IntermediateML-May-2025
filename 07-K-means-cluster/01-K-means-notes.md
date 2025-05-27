### **K-Means Clustering **

K-Means is an unsupervised clustering algorithm used to partition a dataset into $ k $ clusters. The objective is to minimize intra-cluster variance (also called inertia), which is the sum of squared distances between data points and their assigned cluster centroids.

#### **1. Problem Definition**
Given a dataset $ X = \{x_1, x_2, ..., x_n\} $ where $ x_i \in \mathbb{R}^d $, the goal is to partition the data into $ k $ clusters, $ C = \{C_1, C_2, ..., C_k\} $.

#### **2. Steps in K-Means Algorithm**

1. **Initialize $ k $ cluster centroids** $ \mu_1, \mu_2, ..., \mu_k $ randomly.
2. **Assignment Step:** Assign each data point $ x_i $ to the nearest centroid:
   $$
   c_i = \arg\min_{j} ||x_i - \mu_j||^2
   $$
   where $ ||x_i - \mu_j|| $ is the Euclidean distance between $ x_i $ and $ \mu_j $.
3. **Update Step:** Compute new centroids as the mean of all points assigned to that cluster:
   $$
   \mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i
   $$
4. **Repeat Steps 2 and 3** until centroids do not change significantly or a stopping criterion is met.

#### **3. Objective Function**
K-Means minimizes the total within-cluster variance, also called the inertia:

$$
J = \sum_{j=1}^{k} \sum_{x_i \in C_j} ||x_i - \mu_j||^2
$$

where:
- $ J $ is the sum of squared distances of each point to its respective centroid.
- The algorithm iterates to find centroids that minimize $ J $.

---

### **Limitations of K-Means**
- **Sensitive to initialization:** Poor centroid initialization can lead to suboptimal clusters.
- **Fixed number of clusters:** Requires $ k $ to be predefined.
- **Not robust to outliers:** Euclidean distance is sensitive to extreme values.
- **Assumes spherical clusters:** Does not work well with elongated or complex-shaped clusters.



### **Updating Centroids**
For each cluster $ C_j $, the new centroid $ \mu_j $ is calculated as:

$$
\mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i
$$

where:
- $ \mu_j $ is the new centroid of cluster $ j $.
- $ |C_j| $ is the **number of points** in cluster $ j $.
- $ x_i $ are the **data points** in cluster $ j $.

**Intuition**:  
The new centroid is simply the **average** of all points assigned to that cluster.

---

### **Example Calculation**
Let’s assume we have a **2D dataset** and we start with **two clusters**.

#### **Step 1: Initial Data Points**
Consider these six points in a **2D space**:
$$
(2,3), (3,3), (8,8), (9,9), (1,1), (7,8)
$$

Initially, suppose we randomly pick **two centroids**:
- **Centroid 1:** (2,3)
- **Centroid 2:** (9,9)

#### **Step 2: Assign Each Point to the Nearest Centroid**
Using **Euclidean distance**, we assign points to the nearest centroid:

| Data Point | Distance to (2,3) | Distance to (9,9) | Assigned Cluster |
|------------|-------------------|-------------------|------------------|
| (2,3)      | 0                 | 10.63            | Cluster 1        |
| (3,3)      | 1                 | 9.90             | Cluster 1        |
| (8,8)      | 8.49              | 1.41             | Cluster 2        |
| (9,9)      | 10.63             | 0                | Cluster 2        |
| (1,1)      | 2.24              | 11.31            | Cluster 1        |
| (7,8)      | 7.21              | 2.24             | Cluster 2        |

#### **Step 3: Compute New Centroids**
Now, we calculate the new centroid of each cluster by averaging the coordinates of the points assigned to it.

##### **New Centroid for Cluster 1**
Cluster 1 contains: **(2,3), (3,3), (1,1)**  
$$
\text{New centroid} = \left( \frac{2+3+1}{3}, \frac{3+3+1}{3} \right) = \left( \frac{6}{3}, \frac{7}{3} \right) = (2, 2.33)
$$

##### **New Centroid for Cluster 2**
Cluster 2 contains: **(8,8), (9,9), (7,8)**  
$$
\text{New centroid} = \left( \frac{8+9+7}{3}, \frac{8+9+8}{3} \right) = \left( \frac{24}{3}, \frac{25}{3} \right) = (8, 8.33)
$$

#### **Step 4: Repeat Until Convergence**
- The new centroids **(2,2.33) and (8,8.33)** are now used for the next iteration.
- Points are reassigned based on these new centroids.
- The process continues **until centroids stop changing significantly**.

---

### **Key Takeaways**
- The **new centroid is the mean (average) of all points assigned to that cluster**.  
- **Each iteration refines the centroids**, making them more representative of their clusters.  
- The algorithm **converges when centroids stop changing significantly**.  

## Evaluation Metrics 

**Silhouette Score** and **Davies-Bouldin Index (DBI)** are metrics used to evaluate clustering quality, helping determine how well data points are assigned to clusters.

---

### **1. Silhouette Score**
**Definition:**  
The **Silhouette Score** measures how similar an object is to its own cluster compared to other clusters. It provides an assessment of how well-separated the clusters are.

#### **Formula:**
$$
S = \frac{b - a}{\max(a, b)}
$$
where:
- $ a $ = average intra-cluster distance (mean distance to points in the same cluster)
- $ b $ = average nearest-cluster distance (mean distance to the closest different cluster)

#### **Interpretation:**
- **Ranges from -1 to 1**
  - **+1:** Well-clustered, distinct separation between clusters.
  - **0:** Overlapping clusters.
  - **-1:** Misclassified data points (poor clustering).

#### **Use Case:**
- Best for comparing clustering algorithms (e.g., K-Means vs. DBSCAN).
- Helps determine the optimal number of clusters.

---

### **2. Davies-Bouldin Index (DBI)**
**Definition:**  
The **Davies-Bouldin Index** measures the average similarity between each cluster and the most similar cluster. Lower values indicate better clustering.

#### **Formula:**
$$
DBI = \frac{1}{n} \sum_{i=1}^{n} \max_{j \neq i} \left( \frac{s_i + s_j}{d_{ij}} \right)
$$
where:
- $ s_i $ = average distance of cluster $ i $ (cluster dispersion)
- $ d_{ij} $ = distance between cluster centers $ i $ and $ j $

#### **Interpretation:**
- **Lower DBI is better** (good clustering structure).
- **Higher DBI means overlapping clusters** (poor separation).

#### **Use Case:**
- Helps assess clustering compactness and separation.
- Used for selecting the best clustering algorithm.

---

### **Comparison:**
| Metric | Range | Best Value | Measures | Sensitivity to Noise |
|--------|-------|-----------|----------|-----------------------|
| **Silhouette Score** | -1 to 1 | Close to 1 | Cluster separation & cohesion | Moderate |
| **Davies-Bouldin Index** | ≥ 0 | Close to 0 | Cluster similarity | High |

#### **Which One to Use?**
- **Silhouette Score** is more intuitive and widely used.
- **DBI** is useful for validating compact and well-separated clusters.