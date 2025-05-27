### What is ElasticNet Regression?

ElasticNet regression blends **L1 regularization** (from Lasso) and **L2 regularization** (from Ridge) into a single model. Like ordinary least squares (OLS), it minimizes the sum of squared residuals, but it adds a penalty that’s a mix of the absolute values (L1) and squared values (L2) of the coefficients. The cost function is:

```
Cost = Σ(y - ŷ)² + α * [ρ * Σ|β| + (1 - ρ) * Σ(β²)]
```

- **Σ(y - ŷ)²**: The residual sum of squares (same as OLS).
- **α * [ρ * Σ|β| + (1 - ρ) * Σ(β²)]**: The regularization term, where:
  - **α**: Controls the overall strength of the penalty (like in Ridge and Lasso).
  - **ρ** (or `l1_ratio` in scikit-learn): Balances L1 (Lasso) and L2 (Ridge) contributions, ranging from 0 to 1.
  - **Σ|β|** : L1 term (encourages sparsity, as in Lasso).
  - **Σ(β²)**: L2 term (shrinks coefficients, as in Ridge).

- When **ρ = 1**, ElasticNet becomes Lasso (pure L1).
- When **ρ = 0**, it becomes Ridge (pure L2).
- When **0 < ρ < 1**, it’s a mix of both.

This combination allows ElasticNet to perform **feature selection** (like Lasso) while also handling **correlated features** more stably (like Ridge).

### How Does It Impact Model Accuracy?

- **Before (OLS)**: Without regularization, OLS can overfit, especially with multicollinearity or many features, yielding high training accuracy but poor test accuracy.
- **After (ElasticNet)**: By blending L1 and L2 penalties, ElasticNet reduces variance (overfitting) while introducing some bias. It can:
  - Set some coefficients to zero (sparsity from L1).
  - Shrink others without eliminating them (stability from L2).
  - Often improve test accuracy over OLS, Lasso, or Ridge alone when both sparsity and multicollinearity are concerns.
