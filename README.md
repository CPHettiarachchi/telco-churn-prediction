# Telco Customer Churn Prediction - Complete Guide

## Project Overview

This is a complete, production-ready machine learning application designed for predicting customer churn in the telecommunications industry. The project demonstrates key skills required for data science internships.

---

## Project Structure

```
telco-churn-prediction/
│
├── app.py                  # Main Streamlit application (single file)
├── README.md              # This documentation file
└── requirements.txt       # Python dependencies (see below)
```

---

## How to Run the Application

### Prerequisites
- Python 3.8 or higher installed
- Internet connection (for first-time package installation)

### Step 1: Install Dependencies

Create a file named `requirements.txt` with the following content:

```txt
streamlit==1.30.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
```

Then install using:

```bash
pip install -r requirements.txt
```

### Step 2: Run the Application

**IMPORTANT - Use this exact command:**

```bash
python -m streamlit run app.py
```

**DO NOT use:**
- `python app.py`
- `streamlit run app.py` (may not work on all systems)

### Step 3: Access the Application

The terminal will display:
```
You can now view your Streamlit app in your browser.

Local URL: http://localhost:8501
Network URL: http://192.168.x.x:8501
```

Open the **Local URL** in your web browser.

### Step 4: Upload Dataset

1. Download the dataset from [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
2. Click "Browse files" in the sidebar
3. Select the `WA_Fn-UseC_-Telco-Customer-Churn.csv` file

---

## How the Machine Learning Works

### 1️ Data Preprocessing Pipeline

#### Problem: Raw data cannot be directly used for ML models

**Step 1: Drop Non-Predictive Columns**
```python
# customerID is just an identifier, not a feature
data = data.drop('customerID', axis=1)
```

**Step 2: Handle Data Type Issues**
```python
# TotalCharges sometimes has empty strings instead of numbers
data['TotalCharges'] = data['TotalCharges'].replace(' ', np.nan)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
```

**Why this matters:** 
- Prevents "could not convert string to float" errors
- Handles missing data properly
- Ensures all numeric columns are actually numeric

**Step 3: Encode Target Variable**
```python
# Convert Churn from 'Yes'/'No' to 1/0
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
```

**Step 4: One-Hot Encoding for Categorical Features**
```python
# Example: 'gender' column with values ['Male', 'Female']
# Becomes: 'gender_Male' with values [0, 1]

X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
```

**Why drop_first=True?**
- Avoids multicollinearity (redundant information)
- If gender_Male=0, we know it's Female
- Reduces number of features

**Before Encoding:**
| gender | Contract      |
|--------|---------------|
| Male   | Month-to-month|
| Female | One year      |

**After Encoding:**
| gender_Male | Contract_One year | Contract_Two year |
|-------------|-------------------|-------------------|
| 1           | 0                 | 0                 |
| 0           | 1                 | 0                 |

**Step 5: Feature Scaling**
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

**Why scale?**
- MonthlyCharges: range [18-118]
- tenure: range [0-72]
- Without scaling, MonthlyCharges dominates the model
- StandardScaler makes all features have mean=0, std=1

---

### 2️ Model Training

#### Logistic Regression (Baseline Model)
- **Type:** Linear classification model
- **Best for:** Understanding feature relationships
- **Speed:** Very fast
- **Interpretability:** High
- **When to use:** Quick baseline, simple relationships

```python
model = LogisticRegression(random_state=42, max_iter=1000)
```

#### Random Forest (Advanced Model)
- **Type:** Ensemble of decision trees
- **Best for:** Capturing complex patterns
- **Speed:** Slower than logistic regression
- **Interpretability:** Medium (can extract feature importance)
- **When to use:** Better accuracy needed

```python
model = RandomForestClassifier(n_estimators=100, random_state=42)
```

**Parameters explained:**
- `n_estimators=100`: Creates 100 decision trees
- `random_state=42`: Ensures reproducible results
- `max_depth=10`: Prevents overfitting by limiting tree depth

---

### 3️ Making Predictions

When a user inputs new customer data:

**Step 1: Receive Raw Input**
```python
{
    'gender': 'Male',
    'SeniorCitizen': 0,
    'tenure': 12,
    'MonthlyCharges': 65.50,
    ...
}
```

**Step 2: Apply Same Encoding**
```python
# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# One-hot encode (same as training)
input_encoded = pd.get_dummies(input_df, columns=categorical_cols)
```

**Step 3: Ensure Feature Alignment**
```python
# Add missing columns with 0
for col in training_features:
    if col not in input_encoded.columns:
        input_encoded[col] = 0

# Reorder to match training
input_encoded = input_encoded[training_features]
```

**Why this is critical:**
- Model expects features in exact same order as training
- Missing features must be set to 0
- Extra features must be removed

**Step 4: Scale Features**
```python
input_scaled = scaler.transform(input_encoded)
```

**IMPORTANT:** Use `transform()` not `fit_transform()`
- `fit_transform()`: Calculates mean/std from data (training only)
- `transform()`: Uses pre-calculated mean/std (prediction)

**Step 5: Predict**
```python
prediction = model.predict(input_scaled)[0]          # 0 or 1
probability = model.predict_proba(input_scaled)[0][1]  # 0.0 to 1.0
```

**Understanding probability:**
- 0.15 (15%) → Low risk → Customer will likely stay
- 0.55 (55%) → Medium risk → Uncertain
- 0.85 (85%) → High risk → Customer will likely churn

---

## Understanding the Dashboard

### Tab 1: Dashboard
- **Purpose:** Exploratory Data Analysis (EDA)
- **Metrics:** 
  - Total customers in dataset
  - Number who churned
  - Churn rate percentage
- **Visualizations:**
  - Bar chart: Compare churned vs retained
  - Pie chart: Percentage breakdown

### Tab 2: Model Training
- **Confusion Matrix:**
  ```
                Predicted
              No    Yes
  Actual No  [TN]  [FP]
        Yes  [FN]  [TP]
  ```
  - TN (True Negative): Correctly predicted No Churn
  - TP (True Positive): Correctly predicted Churn
  - FP (False Positive): Said Churn, but didn't churn
  - FN (False Negative): Said No Churn, but churned

- **Classification Report:**
  - Precision: Of all predicted churns, how many actually churned?
  - Recall: Of all actual churns, how many did we catch?
  - F1-score: Balance between precision and recall

- **Feature Importance (Random Forest):**
  - Shows which features matter most
  - Example: Contract type, tenure, monthly charges

### Tab 3: Prediction
- **Input Form:** Dynamic fields based on dataset
- **Risk Categorization:**
  - 🟢 Low Risk (< 30%): Customer safe
  - 🟡 Medium Risk (30-70%): Monitor closely
  - 🔴 High Risk (> 70%): Take action now

---

## Sri Lankan Telecom Context

### Why This Project Matters in Sri Lanka

**Local Telecom Landscape:**
- Major players: Dialog, Mobitel, Hutch, Airtel
- High competition → Customer retention critical
- Price-sensitive market → Churn prevention essential

**Business Impact:**
1. **Cost Savings:**
   - Acquiring new customer: Rs. 5,000 - 10,000
   - Retaining existing: Rs. 500 - 2,000
   - 5-20x cheaper to retain!

2. **Revenue Protection:**
   - Average customer lifetime value: Rs. 50,000+
   - Preventing 100 churns = Rs. 5M saved annually

3. **Competitive Advantage:**
   - Predict churn before it happens
   - Proactive retention campaigns
   - Personalized offers

**Real-World Use Cases:**

1. **Dialog's Retention Team:**
   - Use predictions to identify at-risk postpaid customers
   - Offer loyalty bonuses before they switch

2. **Mobitel's Marketing:**
   - Target low-usage customers with data bundle offers
   - Reduce churn in rural areas

3. **Hutch's Customer Service:**
   - Prioritize support for high-risk customers
   - Improve satisfaction scores

**Dataset Adaptation for Sri Lanka:**
- Replace "Contract" with "Bill Payment Method" (eFT, Cash, Card)
- Add "Province" feature (Western, Southern, etc.)
- Include "Language Preference" (Sinhala, Tamil, English)
- Add "Device Type" (Smartphone, Feature phone)

---

## Skills Demonstrated (For Internships)

### Technical Skills
**Python Programming**
- Data manipulation with Pandas
- Numerical computing with NumPy
- Scientific computing with Scikit-learn

**Machine Learning**
- Binary classification
- Model training & evaluation
- Hyperparameter tuning
- Feature engineering

**Data Preprocessing**
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Data type conversions

**Web Application Development**
- Streamlit framework
- Interactive UI/UX design
- Session state management
- File upload handling

**Data Visualization**
- Matplotlib & Seaborn
- Business-oriented charts
- Statistical plots

### Soft Skills
**Problem-Solving**
- End-to-end ML pipeline
- Error handling
- User experience design

**Communication**
- Clear code documentation
- Business insights generation
- Non-technical explanations

**Business Acumen**
- Understanding churn problem
- ROI-focused recommendations
- Industry context awareness

---

## Future Improvements

### 1. Advanced Models
**XGBoost Implementation:**
```python
from xgboost import XGBClassifier

model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
```
**Benefits:**
- 5-10% accuracy improvement
- Better handles imbalanced data
- Feature importance built-in

### 2. Model Explainability
**SHAP (SHapley Additive exPlanations):**
```python
import shap

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Show why customer will churn
shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
```
**Benefits:**
- Explain individual predictions
- Build trust with stakeholders
- Identify key churn drivers

### 3. Hyperparameter Tuning
**Grid Search CV:**
```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5)
grid_search.fit(X_train, y_train)
```

### 4. Deployment Options

**Option 1: Streamlit Cloud (Free)**
```bash
# 1. Push to GitHub
git init
git add .
git commit -m "Initial commit"
git push origin main

# 2. Go to share.streamlit.io
# 3. Connect GitHub repo
# 4. Deploy!
```

**Option 2: Heroku**
```bash
# Create Procfile
web: streamlit run app.py

# Deploy
heroku login
heroku create telco-churn-app
git push heroku main
```

**Option 3: Docker**
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py .

EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### 5. Database Integration
**Store predictions in SQLite:**
```python
import sqlite3

conn = sqlite3.connect('predictions.db')
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE predictions (
        id INTEGER PRIMARY KEY,
        customer_id TEXT,
        prediction INTEGER,
        probability REAL,
        timestamp DATETIME
    )
''')
```

### 6. A/B Testing Framework
**Test retention campaigns:**
- Group A: No intervention (control)
- Group B: Discount offer
- Group C: Loyalty program
- Measure: Actual churn rate after 3 months

### 7. Real-Time Monitoring
**Track model performance:**
```python
# Monitor prediction accuracy over time
# Alert if accuracy drops below threshold
# Retrain model automatically
```

---

## Troubleshooting

### Issue 1: "Could not convert string to float"
**Cause:** TotalCharges has empty strings
**Solution:** Already handled in preprocessing
```python
data['TotalCharges'] = data['TotalCharges'].replace(' ', np.nan)
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'])
```

### Issue 2: "Feature names mismatch"
**Cause:** Input features don't match training features
**Solution:** Align features before prediction
```python
for col in training_features:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[training_features]
```

### Issue 3: "Model not trained"
**Cause:** Trying to predict before training
**Solution:** Check session state
```python
if 'model' not in st.session_state:
    st.warning("Please train model first!")
```

### Issue 4: Streamlit won't run
**Solution:**
```bash
# Use this command
python -m streamlit run app.py

# NOT this
python app.py
```

---

## Learning Resources

### Machine Learning
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Google's ML Crash Course](https://developers.google.com/machine-learning/crash-course)
- [Kaggle Learn](https://www.kaggle.com/learn)

### Streamlit
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [30 Days of Streamlit](https://30days.streamlit.app/)

### Churn Prediction
- [Customer Churn Analysis (Kaggle)](https://www.kaggle.com/code/bandiatindra/telecom-churn-prediction)
- [Churn Prediction Guide](https://towardsdatascience.com/churn-prediction-770d6cb582a5)

---

## Interview Talking Points

When discussing this project in interviews:

1. **Project Overview (30 seconds):**
   > "I built an end-to-end customer churn prediction system using Random Forest and Logistic Regression. The web app allows telecom companies to predict which customers will leave and take proactive retention actions. I handled the complete ML pipeline from data preprocessing to deployment."

2. **Technical Challenges:**
   > "The main challenge was handling the TotalCharges column which had string values mixed with empty strings. I solved this by using pandas' replace and to_numeric functions with error handling."

3. **Business Impact:**
   > "This system could save a telecom company like Dialog millions of rupees annually. Acquiring a new customer costs 5-20x more than retaining an existing one, so predicting churn early is critical."

4. **What You Learned:**
   > "I learned the importance of proper feature engineering, especially one-hot encoding and scaling. I also gained experience building production-ready ML applications with proper error handling and user-friendly interfaces."

5. **Future Improvements:**
   > "I'd love to add SHAP for model explainability, deploy it on the cloud, and integrate real-time data pipelines for continuous learning."

---

## Project Statistics

- **Lines of Code:** ~600
- **Features Engineered:** ~45 (after one-hot encoding)
- **Models Implemented:** 2 (Logistic Regression, Random Forest)
- **Visualizations:** 4 (Distribution, Confusion Matrix, Feature Importance, Risk Cards)
- **Time to Build:** 2-3 days for complete implementation
- **Dependencies:** 6 main libraries

---

## Project Checklist

Before submitting for internship application:

- [ ] Code runs without errors
- [ ] All preprocessing steps documented
- [ ] Model achieves >75% accuracy
- [ ] UI is professional and clean
- [ ] README is comprehensive
- [ ] Comments explain complex sections
- [ ] Requirements.txt is complete
- [ ] Tested with actual dataset
- [ ] Screenshots/demo video prepared
- [ ] GitHub repository is public
- [ ] LinkedIn post written

---


## License

This project is open-source and available for educational purposes.

---

## Acknowledgments

- Dataset: IBM Watson Analytics
- Framework: Streamlit
- ML Library: Scikit-learn
- Inspiration: Real-world telecom churn problems

---

## Author

- Name: Chathura Hettiarachchi  
- Role: Undergraduate Data Science Student  
- University: General Sir John Kotelawala Defence University

---
