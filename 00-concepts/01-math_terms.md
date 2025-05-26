# Essential Statistical Concepts Used in Machine Learning

---

## **1. Mean (Average)**
### **Definition**  
The **mean** is the average value of a dataset.

### **Formula**  
$ 
\mu = \frac{1}{n} \sum_{i=1}^{n} x_i
$
where:
- $ \mu $ = Mean
- $ x_i $ = Data points
- $ n $ = Number of observations

### **Example**
For data: [2, 4, 6, 8],  
$ 
\mu = \frac{2+4+6+8}{4} = 5
$

---

## **2. Median**
### **Definition**  
The **median** is the middle value when data is sorted.

- **Odd dataset**: Middle value.
- **Even dataset**: Average of two middle values.

### **Example**
For **[2, 3, 5, 7, 11]**,  
**Median** = 5 (middle value)

For **[1, 3, 7, 9, 11, 15]**,  
**Median** = $ \frac{7+9}{2} = 8 $

---

## **3. Mode**
### **Definition**  
The **mode** is the most frequently occurring value in a dataset.

### **Example**
For **[1, 2, 2, 3, 3, 3, 4]**,  
**Mode** = 3 (appears most times)

---

## **4. Variance (Measure of Spread)**
### **Definition**  
Variance measures how far data points are from the mean.

### **Formula**
$ 
\sigma^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2
$
where:
- $ \sigma^2 $ = Variance
- $ \mu $ = Mean
- $ x_i $ = Data points
- $ n $ = Number of observations

### **Example**
For **[2, 4, 6, 8]**,  
$ 
\sigma^2 = \frac{(2-5)^2 + (4-5)^2 + (6-5)^2 + (8-5)^2}{4} = 5
$

---

## **5. Standard Deviation**
### **Definition**  
Standard Deviation measures **spread** but in the same units as data.

### **Formula**
$ 
\sigma = \sqrt{\sigma^2} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \mu)^2}
$

### **Example**
For **[2, 4, 6, 8]**,  
$ 
\sigma = \sqrt{5} = 2.236
$

---

## **6. Covariance (Relationship Between Two Variables)**
### **Definition**  
Covariance measures **how two variables move together**.

### **Formula**
$ 
\text{Cov}(X, Y) = \frac{1}{n} \sum_{i=1}^{n} (X_i - \mu_X)(Y_i - \mu_Y)
$
where:
- $ X, Y $ = Two datasets
- $ \mu_X, \mu_Y $ = Their means

### **Interpretation**
- **Positive** → Variables move in **same** direction.
- **Negative** → Variables move in **opposite** direction.
- **Zero** → No relationship.

---

## **7. Correlation (Normalized Covariance)**
### **Definition**  
Correlation measures **strength of relationship** between variables.

### **Formula**
$ 
\rho = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}
$

### **Range**
$ 
-1 \leq \rho \leq 1
$
- **1** → Strong positive correlation.
- **-1** → Strong negative correlation.
- **0** → No correlation.

### **Example**
For **Height vs Weight**,  
- **$ \rho = 0.8 $** → Strong positive relation.
- **$ \rho = -0.5 $** → Weak negative relation.

---

## **8. Probability**
### **Definition**  
Probability quantifies the **likelihood** of an event occurring.

### **Formula**
$ 
P(A) = \frac{\text{Favorable Outcomes}}{\text{Total Outcomes}}
$

### **Example**
- Tossing a coin: $ P(\text{Heads}) = \frac{1}{2} $

---



## **9. Normal Distribution (Gaussian Distribution)**
### **Definition**  
A **bell-shaped curve** where most values cluster around the mean.

### **Formula**
$ 
f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}
$

### **Example**
- **Height distribution in humans** follows a normal distribution.

---

## **10. R-Squared (Model Performance in Regression)**
### **Definition**  
Measures **how well the model explains variance** in data.

### **Formula**
$ 
R^2 = 1 - \frac{SS_{res}}{SS_{tot}}
$
where:
- $ SS_{res} $ = Residual Sum of Squares
- $ SS_{tot} $ = Total Sum of Squares

### **Interpretation**
- $ R^2 = 1 $ → Perfect fit.
- $ R^2 = 0 $ → Model explains nothing.

---

### **Summary Table**
| **Term** | **Usage in ML** |
|----------|----------------|
| Mean, Median, Mode | Data Preprocessing |
| Variance, Standard Deviation | Feature Scaling |
| Correlation, Covariance | Feature Selection |
| Probability | Naive Bayes, Predictive Models |
| Normal Distribution | Assumption for many ML algorithms |
| R-Squared | Regression Model Evaluation |


