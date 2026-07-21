# Impact of AI on Students — Multiclass Classification

This notebook builds and compares several machine learning models to predict a
student's **Burnout Risk Level** (`Low` / `Medium` / `High`) based on their
academic profile and their usage patterns of Generative AI tools.

## Dataset

- **File expected:** `/content/ai_student_impact_dataset.csv` (Google Colab path — update this path if running locally)
- **Rows:** 50,000 students
- **Columns:** 16 total, including:
  - `Student_ID` — dropped before modeling
  - `Major_Category`, `Year_of_Study` — categorical demographics
  - `Pre_Semester_GPA`, `Post_Semester_GPA` — academic performance
  - `Weekly_GenAI_Hours`, `Tool_Diversity`, `Primary_Use_Case`, `Prompt_Engineering_Skill`, `Paid_Subscription` — GenAI usage behavior
  - `Traditional_Study_Hours`, `Perceived_AI_Dependency`, `Institutional_Policy`, `Anxiety_Level_During_Exams`, `Skill_Retention_Score` — study habits and wellbeing indicators
  - `Burnout_Risk_Level` — **target variable** (Medium: 21,144 / Low: 16,369 / High: 12,487)
- No missing values or duplicate rows were found in exploration.

## Workflow

1. **Exploratory Data Analysis (EDA)**
   - `df.info()`, `df.describe()`, null/duplicate checks, class balance check
   - Pairplots of key numerical features colored by burnout risk level

2. **Preprocessing**
   - Dropped `Student_ID`
   - Label-encoded the target (`Burnout_Risk_Level`)
   - One-hot encoded categorical features (`drop_first=True`)
   - Manually pruned redundant one-hot columns down to 14 refined features
   - Train/test split (80/20, `random_state=42`)
   - Standardized features with `StandardScaler` for scale-sensitive models

3. **Models trained & tuned (via `GridSearchCV` where applicable)**
   - Logistic Regression (multinomial)
   - Decision Tree (baseline + tuned, plus a depth-limited comparison tree)
   - Random Forest
   - Gradient Boosting
   - Support Vector Machine (SVC)
   - K-Nearest Neighbors
   - Artificial Neural Network (Keras/TensorFlow `Sequential` MLP with BatchNorm & Dropout, trained with early stopping)

4. **Evaluation**
   - Accuracy, precision, recall, F1-score (weighted/macro)
   - Confusion matrix heatmap
   - Multiclass ROC curves (one-vs-rest, macro-averaged AUC)
   - Training/validation accuracy & loss curves for the ANN

## Results Summary

| Model                     | Accuracy | Weighted F1 |
|---------------------------|:-------:|:-----------:|
| Logistic Regression       | 0.53    | 0.53        |
| Decision Tree (tuned)     | 0.52    | 0.52        |
| Decision Tree (depth=3)   | 0.51    | 0.51        |
| Random Forest             | 0.53    | 0.53        |
| Gradient Boosting         | 0.53    | 0.53        |
| SVM                       | 0.53    | 0.53        |
| K-Nearest Neighbors       | 0.50    | 0.50        |
| ANN (Keras)                | 0.52    | 0.51        |

All models land in a similar ~0.50–0.53 accuracy range on this 3-class problem, suggesting the features have limited separability for this target with these approaches, or that further feature engineering / class-imbalance handling could help.

## Outputs

- `keras_model.h5` — the trained ANN model, saved at the end of the notebook

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
tensorflow / keras
pydotplus (for decision tree visualization; requires Graphviz installed separately)
```

## Notes / Suggestions for Reuse

- The CSV path (`/content/...`) is Colab-specific; change it to a local path if running elsewhere.
- Some cells reference variables defined in earlier (possibly out-of-order) Colab cell executions — run cells top-to-bottom sequentially for reproducibility.
- The `columns_to_drop` feature-selection list in the preprocessing step is manually curated; revisit it if the raw dataset schema changes.
