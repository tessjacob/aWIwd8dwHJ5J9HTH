# Customer Happiness Prediction

### Predicting customer satisfaction from delivery experience survey data

[![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?logo=jupyter)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-EB0028)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainability-9146FF)](https://shap.readthedocs.io/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen)]()
[![Apziva](https://img.shields.io/badge/Apziva-AI%20Residency%20Project-1f2937)](https://www.apziva.com/)

---

## Overview

This project was completed as part of the **Apziva AI Residency Program**, in partnership with a fast-growing logistics and delivery startup. The company wanted to understand what actually drives customer happiness after a delivery experience — using nothing but a short, six-question satisfaction survey.

I was given a de-identified dataset of survey responses and tasked with building a supervised machine learning pipeline that could reliably predict whether a customer is **happy or unhappy**, while also identifying **which service factors matter most**. The end goal wasn't just a model — it was an actionable, data-backed answer to the question:

> *"If you could only fix two or three things about the delivery experience, what should they be?"*

This repository documents the full analytical journey: from raw survey data to a validated, interpretable, production-ready classification model.

---

## Business Problem & Objectives

| Goal | Success Criterion |
|------|-------------------|
| **1. Predict customer happiness** | Classify each customer as happy (1) or unhappy (0) with **≥ 73% accuracy** |
| **2. Identify key satisfaction drivers** | Determine the **minimal set of features** that maximizes predictive power, and flag any feature that can be safely dropped |

This is a **binary classification** problem within a **supervised learning** framework — the historical outcome (happy/unhappy) is known for every customer, and the task is to learn the relationship between survey responses and that outcome.

---

## Dataset

The dataset consists of **126 customer responses** to a six-question satisfaction survey, each answered on a 1–5 scale (low → high agreement).

| Code | Feature | Survey Question |
|------|---------|-----------------|
| `Y`  | **customer_happiness** *(target)* | Overall satisfaction — 0 = unhappy, 1 = happy |
| `X1` | delivery_timeliness               | *"My order was delivered on time"* |
| `X2` | order_accuracy                    | *"Contents of my order was as I expected"* |
| `X3` | product_availability              | *"I ordered everything I wanted to order"* |
| `X4` | price_value_perception            | *"I paid a good price for my order"* |
| `X5` | courier_service_rating            | *"I am satisfied with my courier"* |
| `X6` | app_usability                     | *"The app makes ordering easy for me"* |

---

## Project Workflow

The analysis follows a structured, end-to-end machine learning pipeline:

### Step 1 — Data Loading & Understanding
Imported the raw survey dataset (`data.csv`) and reviewed structure, data types, and value ranges for all six predictors and the target variable.

### Step 2 — Exploratory Data Analysis (EDA)
- Confirmed the dataset contains **126 records** with **no missing values**.
- Identified **16 duplicate records**, which were intentionally retained (not dropped) to preserve the natural response distribution of a small survey dataset.
- Reviewed the **frequency distribution** of every feature (1–5 scale) to understand how customers responded to each question.
- Verified the target variable is **well-balanced**: 56 happy vs. 54 unhappy customers — meaning accuracy is a meaningful metric and the model isn't at risk of trivially predicting the majority class.

### Step 3 — Preprocessing & Data Splitting
- Applied a **stratified train/test split**, preserving the original happy/unhappy ratio in both sets.
- This choice was deliberate: with only 126 observations, an unstratified split risks class imbalance in the test set and an unreliable performance estimate — stratification reduces that sampling bias.

### Step 4 — Model Training & Comparison
Trained and benchmarked **six classification algorithms** to avoid over-relying on any single model's assumptions:

| Model | Accuracy | ROC-AUC |
|-------|----------|---------|
| **Random Forest** | **73.07%** | **0.7976** |
| K-Nearest Neighbors (KNN) | 61.53% | 0.5744 |
| Support Vector Machine (SVM) | 61.53% | 0.7440 |
| Logistic Regression | 61.53% | 0.7500 |
| Naive Bayes | 61.53% | 0.7679 |
| XGBoost | 61.53% | 0.7261 |

*At baseline settings, **Random Forest** was the strongest performer on both accuracy and ROC-AUC — already clearing the 73% target set by the business.*

### Step 4b — Hyperparameter Tuning & Feature Subset Search
Rather than stopping at baseline results, I ran an exhaustive search across **every combination of feature subsets and hyperparameters** for each algorithm, to find each model's true ceiling. Highlights:

| Model | Best Feature Subset | Best Accuracy |
|-------|---------------------|---------------|
| **KNN** (`manhattan`, k=5, uniform weights) | `delivery_timeliness`, `product_availability`, `app_usability` | **84.62%** |
| SVM (`rbf`, C=1) | `delivery_timeliness`, `product_availability` | 80.77% |
| Random Forest (`max_depth=2`, 50 trees) | `delivery_timeliness` | 76.92% |
| Logistic Regression (`C=0.001`, L2) | `delivery_timeliness` | 76.92% |

This step revealed that with the right feature subset and tuning, **a simple KNN classifier outperformed every other model**, reaching **84.61% accuracy** — well above the 73% target.

### Step 5 — Explainability & Feature Importance (SHAP)
To move beyond "which model wins" and answer "**why**," I applied **SHAP (SHapley Additive exPlanations)** analysis to interpret each model's decision-making:

- **`delivery_timeliness`** was ranked the **#1 most important feature by 3 of 4** models analyzed (Logistic Regression, Random Forest, SVM).
- **`order_accuracy`** was the top feature specifically for KNN.
- Averaged across **all six models**, the overall feature importance ranking was:

| Rank | Feature | Avg. Rank Score |
|------|---------|-----------------|
|  1   | `product_availability` | 2.50 |
|  2   | `price_value_perception` | 2.67 |
|  3   | `courier_service_rating` | 3.00 |
| 4    | `order_accuracy` | 3.17 |
| 5    | `delivery_timeliness` | 3.33 |
| 6    | `app_usability` | 4.00 |

> Note: model-specific SHAP results and the aggregated cross-model ranking highlight *delivery_timeliness* and *product_availability* from two different but complementary angles (individual model impact vs. overall consensus rank) — both are consistently in the top tier of influence, while **`app_usability`** and **`courier_service_rating`** are consistently among the least influential.

### Step 6 — Conclusions & Business Recommendations
Synthesized all modeling and explainability results into a final, actionable recommendation (see below).

---

## Key Results

- **Best Model:** K-Nearest Neighbors (KNN) — **84.61% accuracy**, exceeding the 73% target by over **11 percentage points**
- **Minimal predictive feature set:** `delivery_timeliness`, `product_availability`, `app_usability` — a **3-feature model** matches the performance of the full 6-feature model
- **Most influential driver of happiness:** `delivery_timeliness` — "my order was delivered on time"
- **Least influential driver:** `courier_service_rating` — can likely be deprioritized or removed from future surveys without meaningfully hurting predictive power

---

## Business Recommendations

Translating the modeling and SHAP explainability results into concrete, prioritized guidance for product and operations teams:

### Priority 1 — Fix Delivery Timeliness First
`delivery_timeliness` was the #1 most important feature for 3 of 4 models in the SHAP analysis, and is part of the minimal feature subset that achieved the best model accuracy (84.61%).

- Treat on-time delivery as the single highest-leverage lever for customer happiness.
- Invest in delivery SLA monitoring and courier routing/logistics optimization.
- Provide proactive customer communication (real-time tracking, delay alerts) to manage expectations when timeliness slips.
- Tie courier/warehouse KPIs directly to on-time delivery rate, not just delivery completion.

### Priority 2 — Guarantee Product Availability
`product_availability` ranked #1 in the overall cross-model feature ranking and is part of the top-performing minimal feature subset.

- Improve inventory and stock-sync between the app/catalog and real-time warehouse availability.
- Reduce "partial fulfillment" order experiences — prevent out-of-stock items from being orderable, or offer proactive substitutions.
- This is a strong candidate for a fast, high-ROI fix, since it's often a data/systems problem rather than a costly operational overhaul.

### Priority 3 — Reassess Pricing Perception
`price_value_perception` ranked #2 in the overall consensus ranking, ahead of order accuracy and courier service.

- Customers weigh *perceived* value, not just price — review whether pricing communication (fees, discounts, delivery charges) is transparent enough at checkout.
- Test value-reinforcing messaging (e.g., itemized savings, loyalty pricing) rather than assuming price cuts are the only lever.

### Deprioritize — Courier Rating & App Usability
`courier_service_rating` and `app_usability` were consistently the two least influential features across models — `app_usability` had the lowest overall importance rank, and `courier_service_rating` showed minimal SHAP impact in multiple models.

- **Survey design:** these two questions are candidates for removal or reduction in future satisfaction surveys, shortening the survey without losing meaningful predictive signal.
- **Resource allocation:** when choosing between an app UX redesign and a delivery-logistics fix, the data suggests the latter has significantly more impact on customer happiness.

### Strategic Takeaway
The overall pattern suggests customer happiness is driven more by **operational reliability** than by **interface polish or courier interactions**:

> *"Customers don't need a better app — they need their order to show up, complete, on time."*

---

## Methodology & Tools

| Category | Tools / Techniques |
|----------|--------------------|
| **Language** | Python (Jupyter Notebook) |
| **Data Handling** | pandas, NumPy |
| **Modeling** | scikit-learn (Logistic Regression, Random Forest, KNN, SVM, Naive Bayes), XGBoost |
| **Model Explainability** | SHAP |
| **Validation Strategy** | Stratified train/test split, cross-validated hyperparameter & feature-subset search |
| **Evaluation Metrics** | Accuracy, ROC-AUC |

---

## Repository Structure

```
├── data.csv                              # Raw survey dataset (126 responses, 6 features + target)
├── happyCustomersAnalysis_final.ipynb    # Final, polished analysis notebook (start here)
├── happyCustomersAnalysis.ipynb          # Earlier working/exploratory notebook
└── README.md                             # Project documentation (this file)
```

---

## Getting Started

```bash
# Clone the repository
git clone https://github.com/tessjacob/aWIwd8dwHJ5J9HTH.git
cd aWIwd8dwHJ5J9HTH

# Install core dependencies
pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn jupyter

# Launch the notebook
jupyter notebook happyCustomersAnalysis_final.ipynb
```

---

## Notes for Recruiters & Hiring Managers

This project reflects the full lifecycle of a real-world applied machine learning engagement, completed independently as part of Apziva's competitive AI Residency Program. It demonstrates my ability to:

- **Translate a vague business question** ("what makes customers happy?") into a rigorously defined, measurable ML problem with clear success criteria.
- **Work responsibly with limited data** — 126 records is a small sample, and I made deliberate, defensible choices (stratified splitting, retaining duplicates, feature-subset search) to avoid overfitting and misleading conclusions.
- **Avoid single-model tunnel vision** by benchmarking six different algorithm families rather than defaulting to the first one that "worked."
- **Go beyond accuracy** by applying SHAP explainability analysis to make the model's reasoning transparent and business-relevant — because a stakeholder can't act on a black box.
- **Deliver business value, not just metrics** — the final output isn't just "84.61% accuracy," it's a concrete recommendation: *focus on delivery timeliness and product availability, and consider trimming courier-rating questions from future surveys.*

I'm glad to walk through any part of this notebook, the modeling decisions, or the trade-offs I considered — in more detail during an interview.

---

<p align="center">
  <sub>Built as part of the <a href="https://www.apziva.com/">Apziva</a> AI Residency Program</sub>
</p>