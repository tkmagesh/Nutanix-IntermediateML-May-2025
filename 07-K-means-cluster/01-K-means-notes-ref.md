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

### **Evaluation of K-Means Model**
Evaluating a K-Means clustering model involves assessing the quality and stability of the clusters. The most common methods include:

#### **1. Inertia (Within-Cluster Sum of Squares)**
$$
WCSS = \sum_{j=1}^{k} \sum_{x_i \in C_j} ||x_i - \mu_j||^2
$$
- Measures how compact the clusters are.
- Lower values indicate better-defined clusters.
- However, inertia always decreases as $ k $ increases, making it less useful for choosing $ k $.

#### **2. Elbow Method**
- Plots inertia against different values of $ k $.
- The "elbow point" (where the rate of decrease slows down) is considered an optimal choice for $ k $.

#### **3. Silhouette Score**
$$
S = \frac{b - a}{\max(a, b)}
$$
where:
- $ a $ is the average intra-cluster distance (distance between a point and other points in the same cluster).
- $ b $ is the average nearest-cluster distance (distance between a point and the closest different cluster).
- Values range from -1 to 1; higher values indicate better clustering.

#### **4. Davies-Bouldin Index**
$$
DB = \frac{1}{k} \sum_{i=1}^{k} \max_{j \neq i} \frac{s_i + s_j}{d_{ij}}
$$
where:
- $ s_i $ is the average intra-cluster distance.
- $ d_{ij} $ is the inter-cluster distance between centroids $ i $ and $ j $.
- Lower values indicate better clustering.

#### **5. Dunn Index**
$$
DI = \frac{\min_{i,j} d(C_i, C_j)}{\max_k d_k}
$$
where:
- $ d(C_i, C_j) $ is the minimum inter-cluster distance.
- $ d_k $ is the maximum intra-cluster distance.
- Higher values indicate better clustering.

### **Limitations of K-Means**
- **Sensitive to initialization:** Poor centroid initialization can lead to suboptimal clusters.
- **Fixed number of clusters:** Requires $ k $ to be predefined.
- **Not robust to outliers:** Euclidean distance is sensitive to extreme values.
- **Assumes spherical clusters:** Does not work well with elongated or complex-shaped clusters.

### **Step-by-Step Explanation of K-Means Clustering**

K-Means is an **unsupervised machine learning algorithm** that groups similar data points into clusters. The goal is to partition the data into $ k $ clusters where each data point belongs to the nearest cluster centroid.

---

### **Step 1: Choose the Number of Clusters ($ k $)**
- You must first decide how many clusters you want to create. This is a hyperparameter and should be chosen based on the problem.

---

### **Step 2: Initialize $ k $ Cluster Centroids**
- Randomly select $ k $ points from the dataset as initial centroids.
- These centroids represent the center of each cluster.

Example:
If you want to create **3 clusters**, you randomly pick **3 points** from the dataset.

---

### **Step 3: Assign Each Data Point to the Nearest Centroid**
- Calculate the distance between each data point and all centroids.
- Assign each data point to the nearest centroid.

📌 **Common Distance Metric:**
- **Euclidean Distance** (most common): 
  $$
  d(x, \mu) = \sqrt{(x_1 - \mu_1)^2 + (x_2 - \mu_2)^2 + ... + (x_d - \mu_d)^2}
  $$
  where $ x $ is a data point and $ \mu $ is the centroid.

Example:
- If a data point is closer to **Centroid 1** than the other centroids, it gets assigned to **Cluster 1**.

---

### **Step 4: Compute New Centroids**
- Once all points are assigned to clusters, recalculate the centroid of each cluster.
- The new centroid is the **mean** (average) of all points in the cluster.

📌 **Formula to update the centroid:**
$$
\mu_j = \frac{1}{|C_j|} \sum_{x_i \in C_j} x_i
$$
where:
- $ \mu_j $ is the new centroid of cluster $ j $.
- $ |C_j| $ is the number of points in cluster $ j $.
- $ x_i $ are the data points in cluster $ j $.

---

### **Step 5: Repeat Steps 3 and 4 Until Convergence**
- Reassign points to the nearest centroid.
- Recalculate centroids.
- Repeat until **centroids no longer change** significantly or a stopping criterion is met (e.g., a fixed number of iterations).

---

### **Step 6: Final Clustering Result**
- At the end of the iterations, the dataset is divided into $ k $ groups, each with its own centroid.
- Each data point belongs to the cluster whose centroid is closest.

---

### **Example to Illustrate K-Means**
Imagine you have a dataset with the following 6 points in a 2D space:

$$
(2, 3), (3, 3), (8, 8), (9, 9), (1, 1), (7, 8)
$$

If we set $ k = 2 $, the steps would be:

1. **Initialize** two random centroids, say **(2,3) and (9,9)**.
2. **Assign points**:
   - (2,3) is closest to (2,3) → Cluster 1
   - (3,3) is closest to (2,3) → Cluster 1
   - (8,8) is closest to (9,9) → Cluster 2
   - (9,9) is closest to (9,9) → Cluster 2
   - (1,1) is closest to (2,3) → Cluster 1
   - (7,8) is closest to (8,8) → Cluster 2

3. **Update centroids**:
   - New centroid for **Cluster 1**: Average of (2,3), (3,3), and (1,1).
   - New centroid for **Cluster 2**: Average of (8,8), (9,9), and (7,8).
   
4. **Repeat Steps 3 and 4** until centroids stop changing.

---


### **How does a new centroid is identified?**
Once all points are assigned to clusters, the centroid of each cluster is updated by computing the **mean (average) position** of all points in that cluster.

---

### **Mathematical Formula for Updating Centroids**
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

