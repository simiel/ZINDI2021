# ZINDI2021

Working archive for the [UmojaHack Africa 2021 #3: Financial Resilience Challenge (Beginner)](https://zindi.africa/hackathons/umojahack-africa-2021-3-financial-health-challenge-beginner) on Zindi. The task is to predict which individuals, across Africa and elsewhere, are financially resilient: whether they could make a payment in an emergency. The label is binary and the score used in the notebooks is ROC AUC. This repository contains the official starter notebooks and competition files, Samuel Mensah’s follow-up modeling notebooks and Zindi submission files, published reference solutions from other competitors, and a partial set of study notes for Aurélien Géron’s *Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow* (2nd edition).

## Challenge

Each row is one person. `FinancialResilience/Train.csv` has 104,867 rows and 36 columns. `Test.csv` has 44,943 rows and 35 columns (no `target`). `SampleSubmission.csv` has the same 44,943 IDs and an empty `target` column. `VariableDefinitions.csv` documents the fields.

Features are a financial-inclusion survey plus a few demographics:

- `country` (144 countries in the training file) and `region` (eight named regions; 679 training rows have a blank region)
- `population`, `age`
- `owns_mobile`
- survey items `Q1`–`Q28` (there is no `Q13b` or `Q18` in the CSV). Codes are 1 = yes, 2 = no, 3 = don’t know, 4 = refused to answer

Several questions are mostly unanswered. In the training file, `Q7` is blank on 84,226 rows, `Q23` on 67,287, and `Q2` / `Q3` on more than half the rows. The training target counts are 61,011 ones and 43,856 zeros. The starter notebook reads class 1 as people who can make an emergency payment, and describes that class as the majority.

The same four CSVs are copied under `Playground/`. `UHFinancialResilience.zip` is the original bundle of the `FinancialResilience/` folder.

## What is in this repository

### Modeling notebooks

`Untitled3.ipynb` and `Untitled3 (1).ipynb` continue the official Python starter. Both read the CSVs from `/content/drive/MyDrive/FinancialResilience/`. `Untitled3 (1).ipynb` also mounts Google Drive and records that the drive was already mounted. The shared pipeline is:

1. Concatenate train and test (149,810 rows) so encodings stay aligned, then split back using the original training length.
2. Treat `country`, `region`, `owns_mobile`, and every `Q*` column as categorical, and `age` and `population` as numeric.
3. Fill categorical gaps with the column mode and numeric gaps with `9999`, matching the starter.
4. One-hot encode the categorical columns.
5. Hold out 30% of the training rows (`train_test_split`, `random_state=42`).
6. Fit a gradient-boosted classifier, score probabilities with ROC AUC and a classification report at a 0.5 threshold, and write an `ID` / `target` submission.

`Untitled3.ipynb` fits the default `LGBMClassifier`. The saved holdout result is ROC AUC **0.7962515496007383** and accuracy **0.73** (31,461 validation rows). That matches the LightGBM cell in the official Python starter. The notebook also imports `RandomForestClassifier` and calls `cross_val_score` on a name `xgboost` that is not defined in the notebook; that cell has no saved output. It writes `Baseline.csv`.

`Untitled3 (1).ipynb` fits `CatBoostClassifier(learning_rate=0.1, l2_leaf_reg=1500, feature_border_type='MaxLogSum', logging_level='Silent')` on the same split. The training cell has no saved metric. The notebook writes `list.csv`, which is at the repository root. Its Colab log shows CatBoost 0.25 already installed under Python 3.7, and the notebook also runs `pip install catboost` and `pip install ipywidgets`.

### Submission files

The repository root holds 29 submission CSVs. Each has columns `ID` and `target`, 44,943 rows, and a probability in `target`. Names track separate runs: CatBoost (`CAT1.csv`, `CAT3.csv`–`CAT5.csv`, `CATmulti*.csv`, `CATmultileaf*.csv`), LightGBM (`lgbooooooooost*.csv`), XGBoost (`XGB2.csv`), plus `list.csv`, `final.csv`, and `sub*.csv`. The training code that produced every file except `list.csv` is not saved in the notebooks, and no public or private leaderboard score is stored next to them.

`lgbooooooooostdefault.csv` and `sub000.csv` start with the same three probabilities. `subNan2.csv` and `subNan2 (1).csv` use fewer decimal places than the other files.

### Official starter notebooks

`FinancialResilience/Financial_Resilience_Python_StarterNotebook.ipynb` is the Zindi Python starter. It loads the CSVs from the working directory, plots missingness and the target, one-hot encodes categoricals, and compares two defaults on the same 30% split:

| Model | ROC AUC | Accuracy |
| --- | --- | --- |
| `RandomForestClassifier()` | 0.7605856263712535 | 0.70 |
| `LGBMClassifier()` | 0.7962515496007383 | 0.73 |

`FinancialResilience/Financial_Resilience_R_StarterNotebook_ipynb.ipynb` is the R starter (also copied in `Playground/`). It uses tidyverse, fastDummies, caret, and CatBoost R 0.23, with 200 iterations and `Logloss`. The saved confusion matrix reports accuracy **0.7271** (95% CI 0.7221–0.732). It reads `/content/Train.csv` and the matching test, sample, and definition files. `Playground/` does not include the Python starter itself, only its Jupyter checkpoint.

### Reference solutions from other competitors

These notebooks and their READMEs are published write-ups by other competitors, kept here for comparison. Authorship is in those folders. Leaderboard ranks below are claims in those READMEs. The AUC and accuracy figures are outputs saved in the notebooks.

**Financial Resilience, 1st-place notebook.** `UmojaHack-Africa-2021-1th-Place-main/` is the write-up by Saifeddine Azzabi ([@azzabinho001](https://zindi.africa/users/azzabinho001)) and Azer Ksouri ([@ASSAZZIN](https://zindi.africa/users/ASSAZZIN)), team **adatnomerTN**. Their README states rank 1 of 463 and asks for a Tesla P100. The notebook label-encodes `country` and `region`, keeps those codes as categoricals, drops the raw string columns, and runs 5-fold `StratifiedKFold` (split on `country`, `random_state=42`) with GPU CatBoost, LightGBM, and GPU XGBoost. Saved out-of-fold ROC AUC:

| Model | OOF ROC AUC |
| --- | --- |
| CatBoost | 0.8036701848165237 |
| LightGBM | 0.8033790365333908 |
| XGBoost | 0.8020215700198703 |
| Blend `0.5 * CatBoost + 0.4 * LightGBM + 0.1 * XGBoost` | 0.805004547455695 |

The notebook writes that blend to `UmojaHack-Challenge#3-Top1-Solution.csv`. That CSV is not committed here. Data paths are Kaggle paths under `/kaggle/input/umoja2021/`.

**Financial Resilience, companion notebook.** `UmojaHack-Africa-2021-main/Challenge#3 - Financial Resilience/` is an earlier notebook from the same authors. The README says the code is not their final solution and that it “will give you 3rd place,” and it also states rank 1 of 463 for adatnomerTN. It expects a Tesla T4. The model is a 5-fold CatBoost + LightGBM blend with equal weights. Saved OOF ROC AUC is **0.8037272084328954** for CatBoost and **0.8035484429572574** for LightGBM. No blended validation score is stored. It reads `FinancialResilience/Train.csv` and the sibling files, so it expects to be run from this repository’s root. It writes `UmojaHack-Challenge#3-Top3-Solution.csv`, which is not committed here.

**Sendy rider response.** `UmojaHack-Africa-2021-main/Challenge#2 - Sendy - Delivery Rider Response/` is Azer Ksouri’s notebook for [UmojaHack Africa 2021 #2](https://zindi.africa/hackathons/umojahack-africa-2021-2-sendy-challenge-intermediate): predict whether a delivery rider will accept, decline, or ignore an order. The README says the code is not the final solution and that it “will give you 5th place,” and it also states rank 1 of 266. The notebook builds rider–order features (dispatch hour, rider-hour key, rating × number of ratings, bearing from rider to pickup, rating thresholds, and order-level aggregates), then trains GPU `CatBoostClassifier` (`catboost==0.22`, learning rate 0.05, up to 10,000 iterations, early stopping 200) on a stratified 20% holdout. The saved holdout accuracy is **0.6668983154500473**. Processed train and test shapes in the output are `(179867, 36)` and `(76791, 36)`. The Sendy CSVs are not in this repository. The notebook reads `UHSendy/Train.csv`, `Test.csv`, and `Riders.csv` after unpacking a Drive zip.

### Study notes

`handson_ml_2nd-master/` is an incomplete local copy of notes that follow Géron’s book, including Chinese translations of early chapters. Its own README lists the book’s chapter map and pins Python 3.6.9, TensorFlow 2.0.0, and Keras 2.2.4-tf for those notes. Notebooks present here cover chapters 2, 3, 6–10, parts of 12 and 13, and the chapter 6 and 7 exercises. Chapter 1 is included as markdown. That tree is reference material bundled alongside the competition files.

`1_W5ia8bYCS3ZW3LXxU7wSTQ.png` at the repository root is a meme image. It is separate from the charts and metrics saved in the notebooks.

## Tech stack

No dependency file is committed. Libraries are whatever each notebook imports:

- **Python modeling:** pandas, NumPy, matplotlib, seaborn, scikit-learn (`train_test_split`, `roc_auc_score`, `classification_report`, `LabelEncoder`, `StratifiedKFold`), LightGBM, CatBoost, XGBoost in the reference notebooks
- **R starter:** tidyverse, fastDummies, naniar, caret, e1071, CatBoost R 0.23, visdat
- **Runtime in the saved sessions:** Google Colab for the follow-up and Sendy notebooks; Kaggle paths in the 1st-place notebook; a CUDA GPU where a notebook sets CatBoost or XGBoost `task_type` / `tree_method` to GPU

## Layout

```text
FinancialResilience/          competition CSVs and the official Python and R starters
Playground/                   second copy of the CSVs, the R starter, and a Python-starter checkpoint
UHFinancialResilience.zip     zip of the FinancialResilience bundle
Untitled3.ipynb               LightGBM follow-up (Colab)
Untitled3 (1).ipynb           CatBoost follow-up; writes list.csv
*.csv                         29 Zindi submission files (ID, target)
UmojaHack-Africa-2021-1th-Place-main/     published Financial Resilience blend (other authors)
UmojaHack-Africa-2021-main/               published Financial Resilience and Sendy notebooks (other authors)
handson_ml_2nd-master/        partial Hands-On ML study notes
```

## Getting started

There is no install script, environment file, build step, or test suite, and the notebooks do not read environment variables.

Prerequisites:

- Python 3 with Jupyter, or Google Colab, for the Python notebooks
- R with an IR Jupyter kernel for the R starter
- A CUDA GPU only for the reference notebooks that set GPU training (those READMEs ask for a Tesla T4 or P100)
- The Financial Resilience CSVs, already in `FinancialResilience/`

Run the official Python starter with `FinancialResilience/` as the working directory. It calls `pd.read_csv('Train.csv')`, `Test.csv`, `SampleSubmission.csv`, and `VariableDefinitions.csv`.

For `Untitled3.ipynb` and `Untitled3 (1).ipynb`, put those four files at `/content/drive/MyDrive/FinancialResilience/` or change the `read_csv` paths. `Untitled3 (1).ipynb` mounts Drive and installs CatBoost and ipywidgets inside the notebook.

Run `UmojaHack_Challenge_3_Top3_Solution.ipynb` from the repository root so `FinancialResilience/Train.csv` resolves. The 1st-place notebook expects the CSVs at `/kaggle/input/umoja2021/`. The Sendy notebook additionally needs `UHSendy/Train.csv`, `Test.csv`, and `Riders.csv`, which are not included here. The R starter installs its packages in-notebook and reads `/content/Train.csv` and the matching files.

## Project status

Hackathon working archive from UmojaHack Africa 2021, saved as a single commit. It is a notebook-and-submission snapshot. The repository does not store a leaderboard screenshot or a Zindi rank for the 29 submission files.

## What this demonstrates

The follow-up notebooks are a full tabular classification pass on a sparse survey: align train and test before encoding, impute categoricals and numerics separately, one-hot encode high-cardinality country and questionnaire fields, and emit probabilities in Zindi’s `ID` / `target` format. The saved LightGBM holdout matches the starter baseline (ROC AUC 0.7962515496007383, accuracy 0.73). The CatBoost notebook keeps that pipeline and changes the learner: learning rate 0.1, heavy L2 leaf regularization (`l2_leaf_reg=1500`), and `MaxLogSum` borders. The 29 root submission files continue those experiments across CatBoost, LightGBM, and XGBoost runs, including `list.csv` written by the CatBoost notebook.

The published notebooks kept beside that work show the next techniques on the same Financial Resilience data: label-encoded categoricals fed to the booster directly, country-wise stratified folds, GPU CatBoost / LightGBM / XGBoost, and a weighted probability blend whose saved out-of-fold ROC AUC is 0.805004547455695. The Sendy notebook shows a separate feature-engineering pass (time, rider history, and rider-to-pickup bearing) with a recorded holdout accuracy of 0.6668983154500473. Together, the tree is a record of working the beginner challenge and of studying the stronger published blends against the same files.
