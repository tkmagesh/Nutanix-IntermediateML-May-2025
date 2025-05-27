### **L1 Regularization (Lasso)**  

L1 Regularization, also known as **Lasso (Least Absolute Shrinkage and Selection Operator)**, is a technique used to prevent overfitting by **adding a penalty to the absolute values of model weights**. This penalty can shrink some weights to exactly zero, effectively selecting important features.  

---

### **Step 1: Define the Loss Function with L1 Regularization**  
In a standard machine learning model, the loss function (e.g., Mean Squared Error for regression) measures how well the model predicts the target. L1 regularization adds a penalty term:

$$
\text{Loss} = \text{MSE} + \lambda \sum |w_i|
$$

Where:  
- $ MSE = \frac{1}{n} \sum (y_i - \hat{y}_i)^2 $ (for regression models)  
- $ w_i $ = model weights (coefficients)  
- $ \lambda $ = regularization strength (hyperparameter controlling the penalty)  
- $ \sum |w_i| $ = sum of absolute values of weights  

---

### **Step 2: Compute the Gradient Descent Update**  
In normal gradient descent, weights are updated as:

$$
w_i = w_i - \eta \frac{\partial \text{Loss}}{\partial w_i}
$$

For L1 regularization, the gradient of the penalty term $ \lambda |w_i| $ is:  
- $ +\lambda $ when $ w_i > 0 $  
- $ -\lambda $ when $ w_i < 0 $  
- Undefined at $ w_i = 0 $, but usually treated as $ 0 $  

Thus, the weight update rule becomes:

$$
w_i = w_i - \eta (\text{gradient of loss} + \lambda \cdot \text{sign}(w_i))
$$

Where $ \text{sign}(w_i) $ returns $ +1 $ for positive weights and $ -1 $ for negative weights.  

---

### **Step 3: Weights Shrink Toward Zero**  
- If $ \lambda $ is large, some weights will shrink **exactly to zero**, effectively removing features.  
- If $ \lambda $ is small, weights shrink only slightly, keeping most features.  
- If $ \lambda = 0 $, it’s just regular linear regression (no regularization).  

This feature selection property makes L1 regularization useful in **high-dimensional datasets** with irrelevant features.  

---

### **Step 4: Choose the Right λ (Regularization Strength)**  
The value of $ \lambda $ controls the trade-off between:  
- **Low bias, high variance** (small $ \lambda $, less regularization)  
- **High bias, low variance** (large $ \lambda $, stronger regularization, fewer features)  

**Tuning λ:** Use techniques like **cross-validation** to find the optimal value.  

---