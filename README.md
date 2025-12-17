# CausalML Demo 5: Heterogeneous Treatment Effect Estimation

> *"The average causal effect is an average and as such enjoys all the advantages and disadvantages of averages."* – P. W. Holland

This project demonstrates why naive machine learning approaches fail at estimating heterogeneous treatment effects (HTE) and how purpose-built causal methods solve this problem.

## The Core Problem

When treatment effects vary across individuals, we want to estimate **Conditional Average Treatment Effects (CATE)** — how much the treatment helps or hurts different subgroups. The challenge: we never observe both potential outcomes for any individual, so there's no natural loss function to optimize.

Naive ML methods (S-learners, T-learners) tend to learn the wrong signal. They confuse:
- **Prognostic signal** (features predicting baseline risk) for causal moderation
- **Propensity signal** (features predicting treatment assignment) for treatment effect variation

This project shows this failure mode through simulation and demonstrates methods that work.

## Methods Compared

| Method | Approach | When it struggles |
|--------|----------|-------------------|
| **S-learner** | Single model on (X, A) → Y, subtract predictions | Dominated by prognostic signal |
| **T-learner** | Separate models per arm, subtract | High variance with imbalance |
| **X-learner** | Imputes pseudo-outcomes, weights by propensity | Better under imbalance |
| **Causal Forest** | Orthogonalizes nuisance, splits on treatment effect | Needs decent overlap |

## Key Findings

1. **Balanced treatment**: S/T-learners capture baseline outcomes, not treatment effects. Causal Forest and X-learner recover true CATE.

2. **Imbalanced treatment**: S/T-learners degrade further (follow propensity). X-learner maintains performance.

3. **Constant treatment effect**: Naive learners report spurious heterogeneity. Causal methods correctly show minimal variation.

## Data Generating Process

Ten covariates (7 used, 3 noise). The simulation separates:

- **Baseline outcome μ(x)**: Nonlinear function of X1–X6
- **True CATE τ(x)**: Depends on X1, X2, X3 but differently than μ
- **Propensity e(x)**: Controls treatment assignment (balanced ≈ 0.5, imbalanced = extreme)

This setup ensures prognostic and moderating variables overlap but aren't identical — exactly the scenario that trips up naive methods.

## Setup

### Requirements
- Python 3.8+
- Poetry

### Installation

```bash
# Install dependencies
poetry install

# Create Jupyter kernel
poetry run python -m ipykernel install --user --name=lecture-5-env --display-name="CausalML Lecture 5"

# Launch notebook
poetry run jupyter notebook
```

Then select kernel: **Kernel → Change Kernel → "CausalML Lecture 5"**

## Project Structure

```
.
├── lecture_5.ipynb    # Main notebook with simulation and analysis
├── pyproject.toml     # Poetry dependencies
└── README.md
```

## Evaluation Metrics

- **MISE** (Mean Integrated Squared Error): Pointwise error over covariate distribution
- **ATE recovery**: Does τ̂ average to the true ATE?
- **Calibration**: Do high predicted CATEs correspond to actually larger effects?
- **Spurious heterogeneity**: Does the model report variation when τ is constant?

## Dependencies

Core packages:
- `econml` — Causal Forest, X-learner implementations
- `scikit-learn` — Base learners (GBM, Random Forest)
- `plotnine`, `seaborn`, `matplotlib` — Visualization
- `pandas`, `numpy` — Data handling

## References

- Wager & Athey (2018). *Estimation and Inference of Heterogeneous Treatment Effects using Random Forests*
- Künzel et al. (2019). *Metalearners for Estimating Heterogeneous Treatment Effects using Machine Learning*
- Kennedy (2020). *Optimal Doubly Robust Estimation of Heterogeneous Causal Effects*

## Authors

- Giorgio Coppola
- Xiaohan Wu
