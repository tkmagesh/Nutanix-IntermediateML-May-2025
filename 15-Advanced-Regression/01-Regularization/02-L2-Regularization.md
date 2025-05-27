### **L2 Regularization (Ridge Regression)**  

L2 Regularization, also known as **Ridge Regression**, is a technique used to prevent **overfitting** by adding a penalty on the **squared values** of model weights. Unlike L1 regularization, L2 **does not shrink weights to zero**, meaning all features are retained, but it **reduces the magnitude of weights**, making the model more stable.  

---

## **Step 1: Define the Loss Function with L2 Regularization**  
For a normal regression model, the loss function is **Mean Squared Error (MSE)**:

$$
\text{MSE} = \frac{1}{n} \sum (y_i - \hat{y}_i)^2
$$

In **L2 regularization**, we add a penalty on the squared magnitudes of the weights:

$$
\text{Loss} = \text{MSE} + \lambda \sum w_i^2
$$

Where:  
- $ w_i $ = model weights (coefficients)  
- $ \lambda $ = regularization strength (hyperparameter controlling the penalty)  
- $ \sum w_i^2 $ = sum of squared weights  

💡 **Key Idea:** The penalty term forces weights to be smaller, reducing model complexity and preventing overfitting.  

---

## **Step 2: Compute the Gradient Descent Update**  
In standard gradient descent, weights are updated as:

$$
w_i = w_i - \eta \frac{\partial \text{Loss}}{\partial w_i}
$$

For L2 regularization, the gradient of the penalty term $ \lambda w_i^2 $ is:

$$
\frac{\partial}{\partial w_i} (\lambda w_i^2) = 2\lambda w_i
$$

Thus, the weight update rule becomes:

$$
w_i = w_i - \eta (\text{gradient of loss} + 2\lambda w_i)
$$

🔹 This means **L2 regularization reduces weights gradually but does not set them to zero**.  

---

## **Step 3: Weights Shrink, But Not to Zero**  
- If $ \lambda $ is **large**, weights become **very small**, leading to a **simpler model** with reduced variance.  
- If $ \lambda $ is **small**, weights remain **larger**, allowing a more flexible model.  
- If $ \lambda = 0 $, the model is just **linear regression (no regularization).**  

**Difference from L1:** Unlike L1, where some weights become exactly zero (feature selection), **L2 reduces all weights proportionally, but none become zero.**  

---

## **Step 4: Choose the Right λ (Regularization Strength)**  
$ \lambda $ controls the trade-off between:  
- **High variance, low bias** (small $ \lambda $, flexible model)  
- **Low variance, high bias** (large $ \lambda $, simpler model)  