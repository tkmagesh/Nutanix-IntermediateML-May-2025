Let’s explore how hyperparameter tuning can improve the accuracy of a Decision Tree model in machine learning. We shall use the `wine` dataset from scikit-learn, a classification problem with 13 features (e.g., alcohol content, malic acid) and 3 classes (wine types). We’ll compare a default Decision Tree (before tuning) with a tuned version (after tuning) using accuracy as the metric.

### Why Tune Hyperparameters?
A Decision Tree with default settings might grow too deep, overfitting the training data, or be too shallow, underfitting and missing patterns. Tuning parameters like `max_depth`, `min_samples_split`, and `min_samples_leaf` helps balance bias and variance, improving generalization to unseen data.

### Example Using Scikit-Learn

We’ll:
1. Train a default Decision Tree (before tuning).
2. Use GridSearchCV to tune hyperparameters (after tuning).
3. Compare training and test accuracy before and after.

#### Code Example

```python
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the wine dataset
data = load_wine()
X, y = data.data, data.target

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Before Tuning: Default Decision Tree
dt_default = DecisionTreeClassifier(random_state=42)
dt_default.fit(X_train, y_train)

# Predictions
default_train_pred = dt_default.predict(X_train)
default_test_pred = dt_default.predict(X_test)

# Accuracy
default_train_acc = accuracy_score(y_train, default_train_pred)
default_test_acc = accuracy_score(y_test, default_test_pred)

# After Tuning: Hyperparameter Tuning with GridSearchCV
param_grid = {
    'max_depth': [3, 5, 7, None],              # Max tree depth
    'min_samples_split': [2, 5, 10],           # Min samples to split a node
    'min_samples_leaf': [1, 2, 4]              # Min samples in a leaf
}

dt = DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# Best model after tuning
dt_tuned = grid_search.best_estimator_

# Predictions
tuned_train_pred = dt_tuned.predict(X_train)
tuned_test_pred = dt_tuned.predict(X_test)

# Accuracy
tuned_train_acc = accuracy_score(y_train, tuned_train_pred)
tuned_test_acc = accuracy_score(y_test, tuned_test_pred)

# Print results
print("Before Tuning (Default Decision Tree):")
print(f"Training Accuracy: {default_train_acc:.4f}")
print(f"Test Accuracy: {default_test_acc:.4f}\n")

print("After Tuning (Tuned Decision Tree):")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Training Accuracy: {tuned_train_acc:.4f}")
print(f"Test Accuracy: {tuned_test_acc:.4f}")

# Tree depth comparison
print(f"\nDefault Tree Depth: {dt_default.tree_.max_depth}")
print(f"Tuned Tree Depth: {dt_tuned.tree_.max_depth}")
```

#### Sample Output

```
Before Tuning (Default Decision Tree):
Training Accuracy: 1.0000
Test Accuracy: 0.9167

After Tuning (Tuned Decision Tree):
Best Parameters: {'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 2}
Training Accuracy: 0.9929
Test Accuracy: 0.9722

Default Tree Depth: 8
Tuned Tree Depth: 5
```

#### Interpretation

1. **Before Tuning (Default Decision Tree)**:
   - **Training Accuracy (1.0000)**: Perfect fit to the training data (100%), indicating overfitting.
   - **Test Accuracy (0.9167)**: Good but not great—33/36 test samples correct. The tree likely captured noise, hurting generalization.
   - **Tree Depth (8)**: Grew deep, splitting until every training sample was perfectly classified.

2. **After Tuning (Tuned Decision Tree)**:
   - **Training Accuracy (0.9929)**: Slightly lower (99.29%), meaning it didn’t memorize the training data perfectly—less overfitting.
   - **Test Accuracy (0.9722)**: Improved to 97.22% (35/36 correct), a clear jump from 91.67%. Better generalization!
   - **Best Parameters**: `max_depth=5`, `min_samples_split=2`, `min_samples_leaf=1`. The tuned tree is shallower (depth 5 vs. 8), preventing excessive splits.
   - **Tree Depth (5)**: Controlled growth, balancing complexity and accuracy.

#### Why the Improvement?
- **Default Tree**: Overfit by growing too deep (depth 8), fitting noise in the training data (e.g., outliers or quirks in the 13 wine features). This led to a 8.33% drop from training to test accuracy (1.0 to 0.9167).
- **Tuned Tree**: Restricted depth (5) and kept minimal constraints on splitting (e.g., `min_samples_split=2`), capturing key patterns without overfitting. The test accuracy rose by 5.55% (0.9167 to 0.9722), and the train-test gap shrank to 2.07% (0.9929 to 0.9722), showing better consistency.

#### What Changed?
- **max_depth=5**: Limited the tree’s complexity, stopping it from chasing every detail.
- **min_samples_split=2**, **min_samples_leaf=1**: Default-like splitting rules, but the depth cap was the game-changer here.
- The wine dataset’s features (e.g., alcohol, proline) have clear class boundaries, but noise can mislead an unconstrained tree. Tuning found the sweet spot.

### Visualizing the Impact (Optional)
To see the tree structure, you could add `plot_tree(dt_default)` and `plot_tree(dt_tuned)` (as in prior examples), but the depth reduction (8 to 5) already hints at a simpler, more generalizable model.

### Tuning Process
- **GridSearchCV**: Tested 4 × 3 × 3 = 36 combinations across 5-fold cross-validation, picking the one with the highest average CV accuracy.
- **Parameters Tried**:
  - `max_depth`: Controlled overfitting.
  - `min_samples_split/leaf`: Ensured splits were meaningful, though their impact was minor here.

### When Does Tuning Help Most?
- **Small Datasets**: Like wine (178 samples), where overfitting is a risk.
- **Noisy Features**: When some of the 13 features might mislead an untuned tree.
- **Baseline Needed**: To optimize a simple model before jumping to ensembles.

### Conclusion
Hyperparameter tuning transformed this Decision Tree from an overfit memorizer (test accuracy: 91.67%) to a balanced predictor (test accuracy: 97.22%). By capping depth at 5, it avoided noise and boosted generalization on the wine dataset. Tuning is like pruning a real tree—cut the excess, and it thrives!