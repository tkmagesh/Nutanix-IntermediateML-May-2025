### **Logistic Regression**

Logistic Regression is a supervised learning algorithm used for binary classification problems, where the target variable has two possible outcomes (e.g., 0 or 1, True or False, Spam or Not Spam). Unlike linear regression, which predicts continuous values, logistic regression predicts the probability that a given input belongs to a particular class.

#### 1. The Logistic Function (Sigmoid Function)
Logistic Regression uses the **sigmoid function** (also called the logistic function) to transform the linear output into a probability:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

where:
- $ z = w_0 + w_1x_1 + w_2x_2 + \dots + w_nx_n $ (the linear combination of input features)
- $ w_i $ are the model parameters (weights)
- $ x_i $ are the input features
- $ e $ is Euler's number (~2.718)

The output of this function lies in the range (0,1), making it interpretable as a probability.

#### 2. Decision Boundary
To classify an instance, we define a threshold (typically 0.5):

$$
\hat{y} =
\begin{cases} 
1, & \text{if } \sigma(z) \geq 0.5 \\
0, & \text{if } \sigma(z) < 0.5
\end{cases}
$$

This separates the feature space into two regions, each corresponding to a different class.

### Evaluating a Classification Model

When evaluating a logistic regression model, we use various metrics such as **accuracy, precision, recall, F1 score, ROC-AUC and the confusion matrix**. Let's go through each of these with examples.

---

### 1. Accuracy 
**Definition**: Accuracy is the proportion of correctly predicted outcomes out of the total predictions.

#### **Example Scenario**:
Suppose we build a logistic regression model to classify whether an email is spam (1) or not spam (0). Given 100 emails:
- 90 emails are correctly classified (either spam or not spam).
- 10 emails are misclassified.

$$
\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}} = \frac{90}{100} = 90\%
$$

**Limitations**:
- If the dataset is highly imbalanced (e.g., 95% non-spam, 5% spam), a model predicting "not spam" for every email would still achieve 95% accuracy but fail at identifying spam emails.

---

### 2. Precision, Recall, and F1 Score 
These metrics are particularly useful when dealing with imbalanced data.

#### **Example Scenario**:
Consider a **medical test** to detect a disease (1 = Disease, 0 = No Disease). The model produces the following results:

- **True Positives (TP)** = 40 (actual diseased patients correctly identified)
- **False Positives (FP)** = 10 (healthy people incorrectly classified as diseased)
- **False Negatives (FN)** = 30 (diseased people incorrectly classified as healthy)

##### Precision Calculation
$$
\text{Precision} = \frac{\text{TP}}{\text{TP + FP}} = \frac{40}{40 + 10} = \frac{40}{50} = 80\%
$$
**Interpretation**: When the model predicts "disease," it is correct 80% of the time.

##### Recall (Sensitivity) Calculation
$$
\text{Recall} = \frac{\text{TP}}{\text{TP + FN}} = \frac{40}{40 + 30} = \frac{40}{70} = 57.1\%
$$
**Interpretation**: The model correctly identifies 57.1% of all diseased patients.

### 2. Confusion Matrix 
The confusion matrix provides a breakdown of predictions.

#### Example Scenario:
A bank uses a logistic regression model to predict whether a customer will default on a loan.

Default - Positive

No Default - Negative

| Actual \ Predicted | No Default (0) | Default (1) |
|-------------------|--------------|-------------|
| **No Default (0)**  | 50 (TN)  | 5 (FP)  |
| **Default (1)**  | 10 (FN)  | 35 (TP)  |

- **True Negatives (TN) = 50** (Correctly predicted no default)
- **False Positives (FP) = 5** (Wrongly predicted default for a customer who didn’t default)
- **False Negatives (FN) = 10** (Wrongly predicted no default for a customer who actually defaulted)
- **True Positives (TP) = 35** (Correctly predicted default)

**Insights**:
- A high **false negative rate** (FN = 10) could mean the bank is **underestimating loan risk**, leading to financial loss.
- A high **false positive rate** (FP = 5) means some customers were wrongly denied loans.

---

### 3. F1 Score Calculation

F1 Score is the harmonic mean of precision and recall:

$$
F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

$$
F1 = 2 \times \frac{(0.80) \times (0.571)}{0.80 + 0.571} = 66.6\%
$$

**Interpretation**: F1 score balances precision and recall. If missing a diseased patient is costly, recall should be prioritized over precision.

---



### 4. ROC Curve & AUC Example
**Definition**: The Receiver Operating Characteristic (ROC) curve plots **True Positive Rate (Recall) vs. False Positive Rate (FPR)**. The **Area Under Curve (AUC)** measures model performance (higher AUC = better model).

#### Example Scenario:
Suppose we have two loan default prediction models:
- **Model A**: AUC = 0.85 (85% chance of correctly ranking a default case higher than a non-default case)
- **Model B**: AUC = 0.65 (65% chance)

Since **Model A has a higher AUC (0.85 vs. 0.65), it is better at distinguishing defaults from non-defaults**.


### Interpreting the predictions
```
Case-1
Actual - No Default
Predicted - Default
Action: Not approve the loan (for a good customer)
Loss: Loss of interest (oppurtunity loss)
```
```
Case-2
Actual - Default
Predicted - No Default
Action: Approve the loan (for a bad customer)
Loss : Principle (actual loss)
```