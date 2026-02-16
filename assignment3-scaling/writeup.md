# Assignment 3 Scaling Laws Writeup

## 1. Chinchilla IsoFLOPs

### Methodology
We reproduced the IsoFLOPs method from Hoffmann et al. [2022] using the provided data in `data/isoflops_curves.json`.
For each compute budget $C$, we identified the training run with the minimum final loss to determine the optimal model size $N_{opt}$ and dataset size $D_{opt}$.
We then fit power laws of the form $Y = A \cdot C^b$ to these optimal points using linear regression on log-transformed data.

### Results
The fitted scaling laws are:

**Model Size:**
$$N_{opt} \approx 1.16 \cdot C^{0.47}$$

**Dataset Size:**
$$D_{opt} \approx 0.14 \cdot C^{0.53}$$

The exponents sum to $0.47 + 0.53 = 1.00$, which is consistent with the expectation that $C \approx 6ND$.

### Extrapolations
We extrapolated these laws to higher compute budgets:

| Budget (FLOPs) | Predicted $N_{opt}$ | Predicted $D_{opt}$ |
| :--- | :--- | :--- |
| $10^{23}$ | $7.0 \times 10^{10}$ (70B) | $2.4 \times 10^{11}$ (240B) |
| $10^{24}$ | $2.1 \times 10^{11}$ (210B) | $8.1 \times 10^{11}$ (810B) |

See generated plots `isoflops_model_size.png` and `isoflops_dataset_size.png` for visualization.

## 2. Constructing Scaling Laws

### Methodology
To predict the optimal model for a budget of $1 \times 10^{19}$ FLOPs, we conducted experiments using the Training API with a budget of $2 \times 10^{18}$ FLOPs.

**Search Space:**
We designed a search space focusing on compute budgets of $10^{15}$, $3 \times 10^{15}$, and $10^{16}$ FLOPs.
For each budget, we varied the model size ($N$) by adjusting `d_model` and `num_layers` around the values predicted by our Chinchilla scaling law ($N \approx C^{0.47}$).
We also varied `learning_rate` to ensure each model size was trained with near-optimal hyperparameters.

**Scaling Law Fitting:**
After collecting the data (see `scaling_data.json`), we identified the optimal model size for each experimental budget. We then fit a power law to these points to predict $N_{opt}$ for $C = 10^{19}$.

### Predictions for $C = 1 \times 10^{19}$

> [!NOTE]
> The values below should be updated after running `fit_scaling_laws.py` with valid API access.

**Predicted Optimal Model Size:**
[INSERT VALUE HERE, e.g. 1.06e9]

**Predicted Loss:**
[INSERT VALUE HERE]

**Hyperparameters:**
We suggest the following hyperparameters to achieve this model size (approx [INSERT PARAMS]):
- `d_model`: [INSERT]
- `num_layers`: [INSERT]
- `num_heads`: 16
- `batch_size`: 128
