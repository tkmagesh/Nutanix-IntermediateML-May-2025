### **Gradient Descent**
Gradient Descent is an iterative optimization algorithm used to minimize the loss function in **Machine Learning**. 

---

### **Step 1: Define the Problem**
We have a dataset with **input** $ x $ and **output** $ y $. Our goal is to find the **best line** that fits the data:

$$
\hat{y} = mx + b
$$

where:
- $ \hat{y} $ is the predicted value,
- $ m $ (slope) and $ b $ (intercept) are the parameters we need to optimize.

---

### **Step 2: Define the Loss Function (MSE)**
To measure how well our line fits the data, we use the **Mean Squared Error (MSE)**:

$$
MSE = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

where:
- $ n $ = number of data points,
- $ y_i $ = actual value,
- $ \hat{y}_i $ = predicted value $ (mx_i + b) $.

Our goal is to **minimize this error** by adjusting $ m $ and $ b $.

---

### **Step 3: Compute the Gradients**
To find the direction in which to update $ m $ and $ b $, we compute their **partial derivatives** (gradients):

$$
\frac{\partial}{\partial m} MSE = -\frac{2}{n} \sum x_i (y_i - (mx_i + b))
$$

$$
\frac{\partial}{\partial b} MSE = -\frac{2}{n} \sum (y_i - (mx_i + b))
$$

These derivatives tell us **how much the error changes** when we slightly change $ m $ and $ b $.

---

### **Step 4: Update $ m $ and $ b $**
We update $ m $ and $ b $ using **Gradient Descent update rules**:

$$
m = m - \alpha \cdot \frac{\partial}{\partial m} MSE
$$

$$
b = b - \alpha \cdot \frac{\partial}{\partial b} MSE
$$

where:
- $ \alpha $ (learning rate) controls the step size of updates.
- The gradients push $ m $ and $ b $ **in the direction that reduces the error**.

---

### **Step 5: Repeat Until Convergence**
1. **Initialize** $ m $ and $ b $ (e.g., set them to 0 or random values).
2. **Calculate gradients** using the current values of $ m $ and $ b $.
3. **Update** $ m $ and $ b $.
4. **Repeat** steps 2-3 until the error stops decreasing (convergence).

---

### **Example Calculation**
Let's assume:
- We have 3 data points: $ (x, y) = \{(1,2), (2,3), (3,5)\} $.
- We start with $ m = 0 $, $ b = 0 $.
- We choose a learning rate $ \alpha = 0.01 $.

#### **Iteration 1 (Initial values: $ m=0, b=0 $)**
1. Compute predicted values:  
   $$
   \hat{y} = 0 \cdot x + 0 = 0
   $$
   So, predictions: **[0, 0, 0]**

2. Compute gradients:
   $$
   \frac{\partial}{\partial m} = -\frac{2}{3} [(1(2-0)) + (2(3-0)) + (3(5-0))] = -\frac{2}{3} (1\times2 + 2\times3 + 3\times5) = -\frac{2}{3} (2+6+15) = -\frac{2}{3} \times 23 = -15.33
   $$
   $$
   \frac{\partial}{\partial b} = -\frac{2}{3} [(2-0) + (3-0) + (5-0)] = -\frac{2}{3} (2+3+5) = -\frac{2}{3} \times 10 = -6.67
   $$

3. Update parameters:
   $$
   m = 0 - (0.01 \times (-15.33)) = 0.1533
   $$
   $$
   b = 0 - (0.01 \times (-6.67)) = 0.0667
   $$

Now, $ m = 0.1533 $, $ b = 0.0667 $.

#### **Iteration 2**
1. Compute new predictions using updated values.
2. Compute new gradients.
3. Update $ m $ and $ b $.
4. Repeat until convergence.

---

### **When to Stop? (Convergence)**
Gradient Descent stops when:
- The changes in $ m $ and $ b $ are very small.
- The loss function stops decreasing significantly.

---

### **Takeaways**
- **Gradient Descent moves towards the minimum error step by step.**
- **Learning rate ($\alpha$) is crucial:**
  - Too high → Algorithm may overshoot and never converge.
  - Too low → Takes too long to converge.
- **Convergence happens when gradient values approach zero.**
