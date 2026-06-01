# Vertical Farming Door Prediction

## Project Overview

This project applies **machine learning** to a vertical farming environment to predict whether the **door of a farming cube** is **open or closed** using environmental sensor readings.

Vertical farming systems depend heavily on controlled internal conditions such as temperature and humidity. If the cube door remains open at the wrong time, the internal environment may become unstable, which can affect plant growth, energy efficiency, and automation control. This project uses sensor data to build classification models that can help predict the door status and support smarter farming decisions.

---

## Repository Name

```text
vertical-farming-door-prediction
```

---

## GitHub Description

```text
Machine learning project to predict door open/closed status in a vertical farming cube using temperature and humidity sensor data. Includes preprocessing, EDA, feature scaling, and model comparison using Random Forest, SVM, and XGBoost.
```

---

## Problem Statement

The main goal of this project is to build a machine learning model that can predict the **door status** of a vertical farming cube.

The target variable is:

| Target Column | Meaning |
|---|---|
| `Door` | Door status of the vertical farming cube |
| `0` | Door closed |
| `1` | Door open |

The model uses environmental and cube-related features such as:

- Cube ID
- Temperature Layer A
- Temperature Layer B
- Humidity Layer A
- Humidity Layer B

---

## Why This Project Matters

In a vertical farming system, maintaining stable environmental conditions is very important. Temperature and humidity directly affect plant growth, water usage, energy consumption, and crop quality.

Predicting the door status can help in:

- Automating door control
- Maintaining stable internal temperature
- Reducing unnecessary energy loss
- Improving crop-growing conditions
- Supporting smart agriculture and IoT-based farming systems
- Detecting unusual environmental behavior inside the farming cube

---

## Dataset Information

The dataset contains sensor readings from a vertical farming cube.

### Dataset Size

```text
Total records: 32,000
```

### Columns Used

| Column Name | Description |
|---|---|
| `Cube ID` | Unique identifier of the farming cube |
| `Timestamp` | Time of sensor reading |
| `Temperature Layer A` | Temperature reading from layer A |
| `Temperature Layer B` | Temperature reading from layer B |
| `Humidity Layer A` | Humidity reading from layer A |
| `Humidity Layer B` | Humidity reading from layer B |
| `Door` | Target variable representing door status |

### Missing Values

The dataset was checked for missing values, and no missing values were found.

| Column | Missing Values |
|---|---:|
| Cube ID | 0 |
| Timestamp | 0 |
| Temperature Layer A | 0 |
| Temperature Layer B | 0 |
| Humidity Layer A | 0 |
| Humidity Layer B | 0 |
| Door | 0 |

---

## Exploratory Data Analysis

Exploratory Data Analysis was performed to understand the structure, distribution, and relationship between variables.

The EDA included:

- Displaying the first few rows of the dataset
- Checking missing values
- Checking data types
- Generating descriptive statistics
- Plotting a correlation matrix
- Visualizing temperature and humidity distributions
- Creating pair plots for numerical variables
- Comparing temperature and humidity values with door status
- Plotting door status class distribution

### EDA Visualizations Used

| Visualization | Purpose |
|---|---|
| Correlation heatmap | To check relationships among numerical features |
| Histogram with KDE | To view the distribution of temperature and humidity |
| Pair plot | To observe relationships between numerical variables |
| Box plot | To compare sensor readings against door status |
| Bar plot | To check the distribution of open and closed door classes |

---

## Data Preprocessing

The preprocessing steps included:

### 1. Loading the Dataset

The dataset was loaded using Pandas.

```python
import pandas as pd

df = pd.read_csv("verticle_farming_dataset.csv")
```

### 2. Checking Missing Values

```python
df.isnull().sum()
```

No missing values were found.

### 3. Timestamp Conversion

The `Timestamp` column was converted into datetime format and then transformed into Unix timestamp format.

```python
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

df['Timestamp'] = (
    df['Timestamp'] - pd.Timestamp("1970-01-01")
) // pd.Timedelta(seconds=1)
```

### 4. Feature Selection

The selected input features were:

```python
X = df[
    [
        'Cube ID',
        'Temperature Layer A',
        'Temperature Layer B',
        'Humidity Layer A',
        'Humidity Layer B'
    ]
]
```

The target label was:

```python
Y = df['Door']
```

### 5. Feature Scaling

Min-Max Scaling was used to normalize the feature values.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
```

### 6. Train-Test Split

The data was split into training and testing sets.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X_normalized,
    Y,
    test_size=0.2,
    random_state=42
)
```

### Final Data Split

| Dataset | Shape |
|---|---|
| Training features | `(25600, 5)` |
| Testing features | `(6400, 5)` |
| Training labels | `(25600,)` |
| Testing labels | `(6400,)` |

---

## Machine Learning Models Used

Three classification algorithms were applied and compared:

1. Random Forest Classifier
2. Support Vector Machine
3. XGBoost Classifier

---

## Model 1: Random Forest Classifier

Random Forest is an ensemble learning algorithm that builds multiple decision trees and combines their outputs to improve prediction accuracy and reduce overfitting.

### Code Used

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, Y_train)

Y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)
```

### Random Forest Results

| Metric | Score |
|---|---:|
| Accuracy | 0.9920 |
| Precision | 0.9912 |
| Recall | 0.9927 |
| F1-Score | 0.9920 |

---

## Model 2: Support Vector Machine

Support Vector Machine is a supervised learning algorithm that finds the best boundary between classes. It performs well for classification problems where the separation between classes is clear.

### Code Used

```python
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

svm_classifier = SVC()
svm_classifier.fit(X_train, Y_train)

Y_pred_svm = svm_classifier.predict(X_test)
accuracy_svm = accuracy_score(Y_test, Y_pred_svm)
```

### SVM Results

| Metric | Score |
|---|---:|
| Accuracy | 0.9936 |
| Precision | 0.9937 |
| Recall | 0.9934 |
| F1-Score | 0.9935 |

---

## Model 3: XGBoost Classifier

XGBoost is a powerful gradient boosting algorithm that builds models sequentially. Each new model tries to correct the errors made by the previous models.

### Code Used

```python
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

xgb_classifier = XGBClassifier()
xgb_classifier.fit(X_train, Y_train)

Y_pred_xgb = xgb_classifier.predict(X_test)
accuracy_xgb = accuracy_score(Y_test, Y_pred_xgb)
```

### XGBoost Results

| Metric | Score |
|---|---:|
| Accuracy | 0.9923 |
| Precision | 0.9924 |
| Recall | 0.9921 |
| F1-Score | 0.9923 |

---

## Model Comparison

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Random Forest | 0.9920 | 0.9912 | 0.9927 | 0.9920 |
| Support Vector Machine | **0.9936** | **0.9937** | **0.9934** | **0.9935** |
| XGBoost | 0.9923 | 0.9924 | 0.9921 | 0.9923 |

---

## Best Performing Model

The **Support Vector Machine model** achieved the best performance among the three models.

### Best Model

```text
Support Vector Machine
```

### Best Accuracy

```text
99.36%
```

The SVM model produced the highest accuracy, precision, recall, and F1-score, making it the strongest model in this experiment.

---

## Evaluation Metrics

The models were evaluated using the following metrics:

| Metric | Meaning |
|---|---|
| Accuracy | Measures the percentage of total correct predictions |
| Precision | Measures how many predicted positive cases were actually positive |
| Recall | Measures how many actual positive cases were correctly predicted |
| F1-Score | Harmonic mean of precision and recall |
| Confusion Matrix | Shows correct and incorrect predictions for each class |

---

## Confusion Matrix

A confusion matrix was plotted for each model to analyze prediction performance.

The confusion matrix helps identify:

- True positives
- True negatives
- False positives
- False negatives

This is useful because accuracy alone does not always explain how well a classification model performs for each class.

---

## Project Workflow

```text
1. Load dataset
2. Inspect dataset
3. Check missing values
4. Convert timestamp
5. Perform exploratory data analysis
6. Select input features and target variable
7. Normalize features using Min-Max Scaling
8. Split data into training and testing sets
9. Train classification models
10. Evaluate model performance
11. Compare results
12. Select the best-performing model
```

---

## Technologies Used

| Tool / Library | Purpose |
|---|---|
| Python | Main programming language |
| Pandas | Data loading and manipulation |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualizations |
| Scikit-learn | Machine learning models and metrics |
| XGBoost | Gradient boosting classifier |
| Tabulate | Displaying tabular output |
| Jupyter Notebook / Google Colab | Project development environment |

---

## Repository Structure

Recommended GitHub repository structure:

```text
vertical-farming-door-prediction/
│
├── farming_project.ipynb
├── README.md
├── requirements.txt
├── LICENSE
│
├── data/
│   └── verticle_farming_dataset.csv
│
└── images/
    ├── correlation_matrix.png
    ├── distribution_plot.png
    ├── pairplot.png
    ├── boxplot_temperature.png
    ├── boxplot_humidity.png
    └── confusion_matrix.png
```

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/vertical-farming-door-prediction.git
```

### 2. Move Into the Project Folder

```bash
cd vertical-farming-door-prediction
```

### 3. Create a Virtual Environment

For Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

For macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tabulate notebook
```

Or create a `requirements.txt` file and install using:

```bash
pip install -r requirements.txt
```

---

## Suggested requirements.txt

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
tabulate
notebook
```

---

## How to Run the Project

1. Download or clone this repository.
2. Place the dataset file inside the `data/` folder.
3. Open the notebook file:

```bash
jupyter notebook farming_project.ipynb
```

4. Run all cells step by step.
5. View the EDA graphs, model training process, and evaluation results.

If using Google Colab, upload both files:

```text
farming_project.ipynb
verticle_farming_dataset.csv
```

Then update the dataset path if required.

Example:

```python
csv_file_path = "data/verticle_farming_dataset.csv"
```

---

## Key Findings

- The dataset contains **32,000 records**.
- There are **no missing values** in the dataset.
- Temperature and humidity readings are important indicators for predicting door status.
- All three models performed very well.
- Support Vector Machine achieved the best overall performance.
- The best model reached approximately **99.36% accuracy**.

---

## Conclusion

This project demonstrates how machine learning can be used in smart agriculture and vertical farming systems. By using temperature and humidity readings, the model can accurately predict whether the door of a farming cube is open or closed.

The results show that machine learning models can support automation in controlled farming environments. Among the models tested, **Support Vector Machine** performed the best, achieving the highest accuracy, precision, recall, and F1-score.

This type of predictive system can be further improved and integrated into IoT-based farming systems for real-time monitoring and automated control.

---

## Future Improvements

This project can be improved by:

- Adding real-time sensor data
- Using timestamp-based features such as hour, day, and month
- Performing hyperparameter tuning
- Testing more advanced models
- Deploying the model using Flask, FastAPI, or Streamlit
- Creating a dashboard for live monitoring
- Saving the trained model using Pickle or Joblib
- Adding model explainability using SHAP or feature importance
- Testing the model on new unseen farming data
- Connecting the model with IoT devices for real-time automation

---

## Possible Deployment Ideas

The model can be deployed as:

- A Streamlit web app
- A Flask API
- A FastAPI backend service
- An IoT-based prediction system
- A smart farming dashboard
- A real-time monitoring application

---

## Skills Demonstrated

This project demonstrates the following skills:

- Data loading and cleaning
- Exploratory Data Analysis
- Data visualization
- Feature selection
- Feature scaling
- Classification modeling
- Machine learning evaluation
- Model comparison
- Smart agriculture analytics
- Python-based data science workflow

---

## Author

```text
Sameer Hassan Khan
```

---

## License

This project is for educational and research purposes. You may add an open-source license such as the MIT License if you want others to use or contribute to the project.

---

## Acknowledgement

This project is based on a vertical farming dataset and demonstrates the use of machine learning for smart agriculture and environmental monitoring.

