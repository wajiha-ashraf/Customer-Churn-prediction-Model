# Customer Churn Prediction

## 📌 Project Overview

This project uses machine learning to predict whether a customer is likely to churn based on customer behaviour, subscription, usage, payment, and interaction-related features.

The problem is treated as a **binary classification task**:

- `0` → No Churn
- `1` → Churn

The project covers dataset inspection, preprocessing, categorical encoding, feature scaling, Random Forest classification, model evaluation, and visualization.

---

## 📊 Dataset

The dataset used in the notebook is:

`customer_churn_dataset-testing-master.csv`

### Dataset Size

- **Rows:** 64,374
- **Columns:** 12
- **Features used for modelling:** 10
- **Target:** `Churn`

### Original Columns

| Column | Type |
|---|---|
| `CustomerID` | Integer |
| `Age` | Integer |
| `Gender` | Categorical |
| `Tenure` | Integer |
| `Usage Frequency` | Integer |
| `Support Calls` | Integer |
| `Payment Delay` | Integer |
| `Subscription Type` | Categorical |
| `Contract Length` | Categorical |
| `Total Spend` | Integer |
| `Last Interaction` | Integer |
| `Churn` | Integer |

### Target Distribution

The dataset contains:

- **33,881** customers with no churn (`0`) — **52.63%**
- **30,493** customers with churn (`1`) — **47.37%**

### Categorical Values

**Gender**
- Female
- Male

**Subscription Type**
- Basic
- Standard
- Premium

**Contract Length**
- Monthly
- Annual
- Quarterly

---

## 🔍 Data Inspection

The notebook performs:

- Dataset preview using `head()` and `tail()`
- Descriptive statistics using `describe()`
- Shape inspection
- Data type inspection
- Missing-value checking
- Target distribution analysis
- Unique-value inspection for categorical columns

### Missing Values

No missing values were found in the dataset.

---

## 🧹 Data Preprocessing

### 1. Categorical Encoding

The following categorical columns were converted into numerical values using `LabelEncoder`:

```text
Gender
Subscription Type
Contract Length
```

### 2. Removing Customer ID

`CustomerID` was removed because it is an identifier rather than a predictive feature.

```python
df.drop('CustomerID', axis=1, inplace=True)
```

After removing `CustomerID`, the dataset contains **10 input features** plus the target column.

### 3. Feature and Target Separation

```python
X = df.drop('Churn', axis=1)
y = df['Churn']
```

The resulting shapes were:

```text
X: (64374, 10)
y: (64374,)
```

### 4. Train-Test Split

The dataset was split using:

- **80% training data**
- **20% testing data**
- `random_state=42`

Result:

```text
Training samples: 51,499
Testing samples: 12,875
```

### 5. Feature Scaling

`StandardScaler` was applied.

The scaler was fitted on the training data and then used to transform both the training and testing features.

---

## 🤖 Machine Learning Model

A **Random Forest Classifier** was used.

The model configuration was:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

The model was trained on the scaled training data.

---

## 📈 Model Performance

The trained model was evaluated on **12,875 test samples**.

### Accuracy

```text
99.94563106796116%
```

Approximately:

**99.95% Accuracy**

### ROC-AUC

```text
0.9999994069954112
```

### Confusion Matrix

```text
[[6792    1]
 [   6 6076]]
```

This means:

| | Predicted No Churn | Predicted Churn |
|---|---:|---:|
| **Actual No Churn** | 6792 | 1 |
| **Actual Churn** | 6 | 6076 |

### Classification Report

The notebook reports approximately:

| Class | Precision | Recall | F1-score | Support |
|---|---:|---:|---:|---:|
| No Churn (0) | 1.00 | 1.00 | 1.00 | 6,793 |
| Churn (1) | 1.00 | 1.00 | 1.00 | 6,082 |
| Accuracy | | | 1.00 | 12,875 |

Macro and weighted averages are also reported as approximately **1.00**.

---

## 📊 Visualizations

The notebook contains the following visualizations:

### 1. Churn Distribution

A pie chart shows the proportion of customers who churned versus those who did not.

### 2. Feature Importance

A Random Forest feature-importance bar chart is used to visualize the relative importance of the 10 input features.

### 3. Confusion Matrix

A heatmap visualizes actual versus predicted churn classes.

### 4. Age Distribution

A histogram shows the distribution of customer ages.

### 5. ROC Curve

The ROC curve is plotted using the predicted churn probabilities.

The notebook reports:

```text
AUC Score: 0.9999994069954112
```

---

## 🛠️ Technologies & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

### Scikit-learn Components Used

- `train_test_split`
- `LabelEncoder`
- `StandardScaler`
- `RandomForestClassifier`
- `classification_report`
- `confusion_matrix`
- `roc_auc_score`
- `roc_curve`

---

## 🔄 Project Workflow

```text
Raw Dataset
     ↓
Data Inspection
     ↓
Missing Value Check
     ↓
Categorical Encoding
     ↓
Remove CustomerID
     ↓
Feature / Target Separation
     ↓
Train-Test Split
     ↓
Standard Scaling
     ↓
Random Forest Classifier
     ↓
Predictions
     ↓
Model Evaluation
     ↓
Visualizations
```

---

## 📂 Project Structure

```text
Customer-Churn-Prediction/
│
├── customer_churn_dataset-testing-master.csv
├── main.ipynb
└── README.md
```

---

## 🚀 How to Run

### 1. Install the required libraries

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

### 2. Open the notebook

```bash
jupyter notebook
```

### 3. Run `main.ipynb`

Make sure the dataset file is available in the project directory.

---

## 🎯 Project Objective

The objective of this project is to build a binary classification model that predicts customer churn using customer demographic, usage, support, payment, subscription, and interaction-related information.

The project demonstrates a complete machine learning workflow from **data inspection and preprocessing to model training, evaluation, and visualization**.

---

## 👩‍💻 Author

**Wajiha Ashraf**

BS Computer Science | AI/ML & Data Science Enthusiast
