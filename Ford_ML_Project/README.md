# Used Ford Price Prediction

## Project question

**Can we estimate the asking price (GBP) of a used Ford from its model, year, transmission, mileage, fuel type, tax, fuel economy, and engine size?**

This project is a supervised regression workflow that cleans the provided Ford listings, explores patterns, compares regression models, and reports an honest final holdout evaluation. It is an educational estimate of listing prices, not a transaction-price valuation.

## Contents

- `ford.csv` — original supplied dataset (17,966 rows, 9 columns).
- `ford_cleaned.csv` — cleaned data export created by the notebook and used for all EDA/modeling.
- `ford_used_car_price_prediction.ipynb` — the single end-to-end notebook, following the supplied blueprint.
- `README.md` — this project guide.

## Data preparation

The notebook verifies the source, removes exact duplicate rows, trims category whitespace, and investigates the one year-2060 record. It excludes that record because it is outside the source's observed 1996–2020 listing period and the source provides no evidence for a corrected year. It retains the row with zero engine size but marks that impossible measurement as missing; the model pipeline imputes it from training-fold medians. Other nonpositive price, negative mileage/tax, and missingness checks are reported before cleaning. The notebook exports `ford_cleaned.csv` beside itself, rereads it, and uses that export for every downstream step.

## How to run

1. Activate the **sample1** environment and launch Jupyter from it. If Jupyter does not list the environment as a kernel, register it once with `python -m ipykernel install --user --name sample1 --display-name "Python (sample1)"`.
2. Open `ford_used_car_price_prediction.ipynb`.
3. Run all cells from top to bottom. Keep `ford.csv` in the same folder.

Required libraries: Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, and Jupyter. All plots use Matplotlib and Seaborn. The notebook uses a fixed random seed (42), a 20% holdout, and five-fold shuffled cross-validation on training data only.

## Evaluation

RMSE is the primary training-CV selection metric. A mean baseline and Ridge, random forest, extra trees, and histogram gradient boosting models are compared with consistent preprocessing. The selected pipeline is evaluated once on the reserved test split; MAE and RMSE are expressed in GBP, with R² and residual plots for context. The results section in the notebook is populated by executing all cells.

## Limitations

The target is listing price, which can differ from the sale price. Condition, trim, location, ownership history, and listing identity are not included. Random splitting may place similar listings in both sets. The year-2060 anomaly is excluded rather than guessed at. These model patterns are associations and should not be interpreted as causal.
