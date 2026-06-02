# ML Sales Forecasting

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24%2B-orange?logo=scikit-learn)
![pandas](https://img.shields.io/badge/pandas-1.5%2B-150458?logo=pandas)
![License](https://img.shields.io/badge/License-MIT-green)

A **linear regression pipeline** that forecasts mobile and tablet sales for an online retailer. The project covers the full ML workflow: data loading, exploratory data analysis, feature engineering, model building with three progressively complex models, performance evaluation, and auto-generated timestamped reports in both **HTML** and **Markdown** formats.

---

## Table of Contents

1. [Business Context](#business-context)
2. [Project Architecture](#project-architecture)
3. [Data Dictionary](#data-dictionary)
4. [Libraries & Dependencies](#libraries--dependencies)
5. [Implementation Pipeline](#implementation-pipeline)
6. [Models](#models)
7. [EDA Highlights](#eda-highlights)
8. [Plots Catalogue](#plots-catalogue)
9. [Report Generation](#report-generation)
10. [How to Run](#how-to-run)
11. [Results Summary](#results-summary)

---

## Business Context

An online retailer selling mobiles and tablets faces challenges in **inventory management** and **marketing spend allocation**. Accurately forecasting sales enables:

- Reducing stockouts and overstock situations
- Allocating advertising budgets efficiently
- Gaining competitive advantage through data-driven decisions

As part of this analysis, historical sales data is used alongside pricing, promotions, and customer engagement features to build a predictive linear regression model.

---

## Project Architecture

```
ml-sales-forecasting/
│
├── main.py                    # Entry point — runs the full pipeline
├── requirements.txt           # Python dependencies
├── .gitignore
│
├── data/
│   └── Sales.csv              # Dataset (3,000 rows × 10 columns)
│
├── src/
│   ├── data_generator.py      # Synthetic data generator (fallback if CSV missing)
│   ├── eda.py                 # Exploratory data analysis — saves 6 plots
│   ├── models.py              # Builds all 3 linear regression models
│   ├── evaluation.py          # Computes RMSE / MAE / MAPE, saves comparison charts
│   └── reporter.py            # Generates timestamped HTML + Markdown reports
│
└── outputs/
    ├── plots/                 # EDA and model fit plots (7 PNG files)
    ├── model_comparison.png   # Bar chart comparing all 3 models
    ├── residual_plot.png      # Residuals vs predicted for best model
    ├── train_performance.csv  # Training metrics for all models
    ├── test_performance.csv   # Test metrics for all models
    ├── report_<timestamp>.html  # Self-contained HTML report (embedded images)
    └── report_<timestamp>.md    # Markdown report (relative image links)
```

### Module Responsibilities

| Module | Responsibility |
|:-------|:--------------|
| `data_generator.py` | Generates 3,000 synthetic rows using the best-model coefficients + Gaussian noise. Used as a fallback when `Sales.csv` is absent. |
| `eda.py` | Produces histograms, box plots, count plots, correlation heatmap, and Sales vs. categorical breakdowns. All saved to `outputs/plots/`. |
| `models.py` | Encodes `Popularity` (ordinal label encoding) and `Region` (one-hot, East as baseline), trains 3 `sklearn` `LinearRegression` models, saves the simple-LR best-fit scatter. |
| `evaluation.py` | Computes RMSE, MAE, MAPE for train and test sets across all 3 models; saves comparison bar chart and residual plot; returns DataFrames to the reporter. |
| `reporter.py` | Reads performance DataFrames + plot images; writes a fully self-contained `.html` (base64-embedded images) and a `.md` (relative image links), both timestamped. |

---

## Data Dictionary

| # | Feature | Type | Description |
|---|:--------|:-----|:------------|
| 1 | **Advertising Expenditure** | Numeric | Amount spent on ads ($). Range: $97 – $802 |
| 2 | **Campaign Engagement Score** | Numeric | Social media engagement score (likes, comments, shares) |
| 3 | **Discount Percentage** | Numeric | Average discount offered (%). Range: 0 – 40.5% |
| 4 | **Average Customer Rating** | Numeric | Customer product rating. Range: 3.65 – 4.87 |
| 5 | **Product Price** | Numeric | Product price ($). Range: $10 – $2,001 |
| 6 | **Return Rate** | Numeric | Average return rate post-delivery. Range: 0.03 – 4.53 |
| 7 | **Length of Product Description** | Numeric | Word count of product listing. Range: 42 – 496 words |
| 8 | **Region** | Categorical | North / South / East / West |
| 9 | **Popularity** | Categorical (Ordinal) | Very Low / Low / Moderate / High / Very High |
| 10 | **Sales** *(target)* | Numeric | Total sales in dollars. Range: $3,258 – $34,421 |

**Dataset:** 3,000 rows · 8 numerical features · 2 categorical features · 0 missing values · 0 duplicates

---

## Libraries & Dependencies

| Library | Version | Purpose |
|:--------|:--------|:--------|
| `pandas` | ≥ 1.5.0 | Data loading, manipulation, DataFrame operations |
| `numpy` | ≥ 1.21.0 | Numerical computing, array operations |
| `matplotlib` | ≥ 3.5.0 | Base plotting library |
| `seaborn` | ≥ 0.12.0 | Statistical visualisations (histplots, heatmaps, boxplots) |
| `scikit-learn` | ≥ 0.24.0 | `LinearRegression`, `train_test_split`, RMSE/MAE/MAPE metrics |

Install all dependencies:

```bash
pip install -r requirements.txt
```

---

## Implementation Pipeline

When `main.py` is executed, the following steps run in order:

```
1. Data Loading
   └─ Reads data/Sales.csv (auto-generates via data_generator.py if missing)

2. Data Overview
   └─ Shape, head, describe(), missing values, duplicate check

3. Exploratory Data Analysis  [src/eda.py]
   ├─ Univariate:  histograms + box plots for all 8 numeric features
   ├─ Categorical: count plots for Region and Popularity
   └─ Bivariate:   correlation heatmap, Sales vs Popularity, Sales vs Region

4. Feature Engineering  [src/models.py]
   ├─ Ordinal encoding:  Popularity  →  {Very Low:1, Low:2, ..., Very High:5}
   └─ One-hot encoding:  Region      →  Region_North, Region_South, Region_West
                                        (East dropped as baseline)

5. Train / Test Split
   └─ 80% train / 20% test  (random_state=42, reproducible)

6. Model Building  [src/models.py]
   ├─ Model 1:  Simple LR       — Advertising Expenditure only
   ├─ Model 2:  Multiple LR     — Adv. Exp. + Discount % + Product Price
   └─ Model 3:  Multiple LR     — All 11 features (post-encoding)

7. Evaluation  [src/evaluation.py]
   └─ RMSE, MAE, MAPE on both train and test sets for all 3 models

8. Report Generation  [src/reporter.py]
   ├─ outputs/report_<YYYY-MM-DD_HH-MM-SS>.html   (self-contained, ~527 KB)
   └─ outputs/report_<YYYY-MM-DD_HH-MM-SS>.md     (~4 KB, relative image links)
```

---

## Models

### Model 1 — Simple Linear Regression

**Features used:** `Advertising Expenditure`

```
Sales = (28.6675) × Advertising Expenditure  +  6435.8913
```

Advertising Expenditure alone explains a significant portion of variance, confirming it as the strongest single predictor.

---

### Model 2 — Multiple Linear Regression (3 Features)

**Features used:** `Advertising Expenditure`, `Discount Percentage`, `Product Price`

```
Sales = (38.4366) × Advertising Expenditure
      + (-3.5748) × Discount Percentage
      + (-8.4560) × Product Price
      + 12731.1316
```

Adding discount and price reduces RMSE by ~24% compared to the simple model.

---

### Model 3 — Multiple Linear Regression (All Features) ✅ Best

**Features used:** All 11 encoded features

```
Sales = (36.8004) × Advertising Expenditure
      + (20.7776) × Campaign Engagement Score
      + (-6.9185) × Discount Percentage
      + (2890.2781) × Average Customer Rating
      + (-8.6700) × Product Price
      + (-444.7877) × Return Rate
      + (-0.0509) × Length of Product Description
      + (146.8423) × Popularity
      + (-6.9811) × Region_North
      + (52.8818) × Region_South
      + (189.1674) × Region_West
      + 469.5990
```

**Key coefficient insights:**
- `Average Customer Rating` has the largest positive impact (+$2,890 per rating point)
- `Return Rate` has the largest negative impact (−$445 per unit increase)
- `Popularity` adds ~$147 per level increase (Very Low → Very High = +$588)
- West region products show ~$189 higher sales than East (baseline)

---

## EDA Highlights

| Finding | Detail |
|:--------|:-------|
| Distribution | Sales, Advertising Expenditure, Discount %, and Product Price are left-skewed |
| Outliers | Present in all features; Return Rate and Product Price most affected |
| Strongest predictor | Advertising Expenditure has the highest positive correlation with Sales |
| Negative correlations | Product Price and Return Rate are negatively correlated with Sales |
| Regional uniformity | Sales are largely uniform across North / South / East / West |
| Popularity impact | Unpopular products (Very Low) show noticeably lower median sales |
| Multicollinearity | Advertising Expenditure and Campaign Engagement Score are positively correlated |
| Rating vs Return | Average Customer Rating and Return Rate are negatively correlated |

---

## Plots Catalogue

All plots are saved to `outputs/plots/` on each run.

| File | Description |
|:-----|:-----------|
| `histograms.png` | Distribution of all 8 numerical features |
| `boxplots.png` | Outlier detection for all 8 numerical features |
| `categorical_counts.png` | Count plots for Region and Popularity |
| `correlation_heatmap.png` | Pearson correlation matrix for all numerical features |
| `sales_by_popularity.png` | Box plot of Sales across 5 popularity levels |
| `sales_by_region.png` | Box plot of Sales across 4 regions |
| `simple_lr_fit.png` | Scatter plot with best-fit line (Model 1) |
| `model_comparison.png` | Bar chart of RMSE / MAE / MAPE for all 3 models (test set) |
| `residual_plot.png` | Residuals vs. predicted values for the best model |

---

## Report Generation

Every run of `main.py` creates two timestamped report files in `outputs/`:

### HTML Report (`report_<timestamp>.html`)

- **Fully self-contained** — all 9 plots are base64-encoded and embedded inline
- Open in any browser without needing the `outputs/plots/` folder
- Includes: dataset stats table, all EDA plots in a responsive grid, model equations, colour-coded performance tables (best model highlighted in green), comparison chart, residual plot, and a summary box
- Typical file size: ~527 KB

### Markdown Report (`report_<timestamp>.md`)

- Lightweight (~4 KB) with relative image links (`plots/xxx.png`)
- Renders correctly in VS Code Preview, GitHub, Obsidian, and any GFM-compatible viewer
- Contains the same sections as the HTML report in standard Markdown table format
- Ideal for committing alongside code for readable diffs

> Reports are **excluded from git** via `.gitignore` (`outputs/report_*.html`, `outputs/report_*.md`) since they are regenerated on every run with a new timestamp.

---

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/rajaembees/ml-sales-forecasting.git
cd ml-sales-forecasting
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the pipeline

```bash
python main.py
```

The script will:
- Use `data/Sales.csv` if present, otherwise auto-generate 3,000 synthetic rows
- Save all plots to `outputs/plots/`
- Print training and test metrics to the console
- Write a timestamped HTML + Markdown report to `outputs/`

### Branch Strategy

| Branch | Purpose |
|:-------|:--------|
| `main` | Stable, production-ready code |
| `develop` | Active development; feature branches merge here first |

---

## Results Summary

### Test Set Performance

| Model | RMSE | MAE | MAPE |
|:------|-----:|----:|-----:|
| Simple LR (Advertising Expenditure only) | $2,857 | $2,226 | 9.96% |
| Multiple LR (Adv. Exp. + Discount % + Price) | $2,179 | $1,741 | 7.66% |
| **Multiple LR (All Features)** ✅ | **$1,908** | **$1,558** | **6.76%** |

### Training Set Performance

| Model | RMSE | MAE | MAPE |
|:------|-----:|----:|-----:|
| Simple LR (Advertising Expenditure only) | $2,753 | $2,208 | 9.65% |
| Multiple LR (Adv. Exp. + Discount % + Price) | $2,170 | $1,757 | 7.64% |
| **Multiple LR (All Features)** ✅ | **$1,941** | **$1,565** | **6.79%** |

**The full-feature model achieves a 33% improvement in RMSE over the simple model**, with train and test metrics closely aligned — no evidence of overfitting.

---

*Built with reference to the [Hands-on Linear Regression](./Hands_on_Linear_Regression.ipynb) notebook.*
