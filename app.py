"""
Telco Customer Churn Prediction System
========================================
A complete machine learning application for predicting customer churn in telecom industry.

Author: Data Science Student
Purpose: Internship Portfolio Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Telco Churn Predictor",
    page_icon="📞",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)


def load_and_validate_data(uploaded_file):
    """
    Load CSV file and validate required columns
    
    Parameters:
    -----------
    uploaded_file : UploadedFile
        Streamlit uploaded file object
        
    Returns:
    --------
    pd.DataFrame or None
        Loaded dataframe if valid, None otherwise
    """
    try:
        df = pd.read_csv(uploaded_file)
        
        # Check if Churn column exists
        if 'Churn' not in df.columns:
            st.error("❌ Dataset must contain a 'Churn' column!")
            return None
            
        return df
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None


def preprocess_data(df):
    """
    Complete data preprocessing pipeline
    
    Steps:
    1. Drop customerID (non-predictive)
    2. Handle missing values
    3. Convert TotalCharges to numeric
    4. Encode target variable (Churn)
    5. Separate features and target
    
    Parameters:
    -----------
    df : pd.DataFrame
        Raw dataset
        
    Returns:
    --------
    tuple: (X, y, feature_names, categorical_columns, numeric_columns)
    """
    # Create a copy to avoid modifying original data
    data = df.copy()
    
    # Step 1: Drop customerID if it exists (it's just an identifier, not a feature)
    if 'customerID' in data.columns:
        data = data.drop('customerID', axis=1)
    
    # Step 2: Handle TotalCharges column (sometimes stored as string with spaces)
    if 'TotalCharges' in data.columns:
        # Replace empty strings with NaN
        data['TotalCharges'] = data['TotalCharges'].replace(' ', np.nan)
        # Convert to numeric
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        # Fill missing values with median
        data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)
    
    # Step 3: Encode target variable
    # Convert Churn from Yes/No to 1/0
    data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
    
    # Step 4: Separate features (X) and target (y)
    X = data.drop('Churn', axis=1)
    y = data['Churn']
    
    # Step 5: Identify column types
    categorical_columns = X.select_dtypes(include=['object']).columns.tolist()
    numeric_columns = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Step 6: One-Hot Encode categorical variables
    # This converts categorical features into binary columns
    X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=True)
    
    feature_names = X_encoded.columns.tolist()
    
    return X_encoded, y, feature_names, categorical_columns, numeric_columns


def train_model(X_train, X_test, y_train, y_test, model_type='Logistic Regression'):
    """
    Train selected ML model with optimized hyperparameters for best performance
    
    Parameters:
    -----------
    X_train, X_test : pd.DataFrame
        Training and testing features
    y_train, y_test : pd.Series
        Training and testing labels
    model_type : str
        Type of model to train
        
    Returns:
    --------
    tuple: (model, scaler, accuracy, y_pred, y_pred_proba, roc_auc)
    """
    # Step 1: Feature Scaling (use StandardScaler for consistency)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 2: Initialize model with optimized hyperparameters
    if model_type == 'Logistic Regression':
        model = LogisticRegression(
            random_state=42,
            max_iter=2000,
            C=0.1,  # Regularization
            solver='lbfgs',
            class_weight='balanced'  # Handle imbalanced data
        )
        
    elif model_type == 'Random Forest':
        model = RandomForestClassifier(
            n_estimators=200,  # More trees for better accuracy
            random_state=42,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight='balanced',
            n_jobs=-1  # Use all CPU cores
        )
        
    elif model_type == 'Gradient Boosting':
        model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )
        
    elif model_type == 'XGBoost':
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                min_child_weight=1,
                gamma=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                objective='binary:logistic',
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False,
                scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1])
            )
        except ImportError:
            st.warning("⚠️ XGBoost not installed. Using Gradient Boosting instead.")
            st.info("To install XGBoost: pip install xgboost")
            model = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                random_state=42
            )
    
    # Step 3: Train the model
    model.fit(X_train_scaled, y_train)
    
    # Step 4: Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Step 5: Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    return model, scaler, accuracy, y_pred, y_pred_proba, roc_auc


def plot_churn_distribution(df):
    """
    Visualize churn distribution in the dataset
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    
    # Count plot
    churn_counts = df['Churn'].value_counts()
    ax[0].bar(['No Churn', 'Churn'], churn_counts.values, color=['#2ecc71', '#e74c3c'])
    ax[0].set_ylabel('Number of Customers')
    ax[0].set_title('Churn Distribution')
    ax[0].grid(axis='y', alpha=0.3)
    
    # Pie chart
    colors = ['#2ecc71', '#e74c3c']
    ax[1].pie(churn_counts.values, labels=['No Churn', 'Churn'], autopct='%1.1f%%', 
              colors=colors, startangle=90)
    ax[1].set_title('Churn Percentage')
    
    plt.tight_layout()
    return fig


def plot_confusion_matrix(y_test, y_pred):
    """
    Visualize confusion matrix
    """
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title('Confusion Matrix')
    return fig


def plot_roc_curve(y_test, y_pred_proba):
    """
    Visualize ROC curve
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
            label='Random Classifier')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Model Performance')
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    return fig


def plot_feature_importance(model, feature_names, top_n=15):
    """
    Plot feature importance for tree-based and boosting models
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importances[indices], color='skyblue')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Most Important Features')
        plt.tight_layout()
        return fig
    return None


def get_risk_level(probability):
    """
    Categorize churn probability into risk levels
    
    Parameters:
    -----------
    probability : float
        Churn probability (0-1)
        
    Returns:
    --------
    tuple: (risk_level, color)
    """
    if probability < 0.3:
        return "🟢 Low Risk", "#2ecc71"
    elif probability < 0.7:
        return "🟡 Medium Risk", "#f39c12"
    else:
        return "🔴 High Risk", "#e74c3c"


# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Header
st.markdown('<h1 class="main-header">📞 Telco Customer Churn Prediction System</h1>', 
            unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/phone.png", width=80)
    st.title("⚙️ Configuration")
    
    # File uploader
    st.subheader("📁 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload Telco Customer Churn CSV",
        type=['csv'],
        help="Upload the Telco Customer Churn dataset from Kaggle"
    )
    
    # Model selection
    st.subheader("🤖 Model Selection")
    model_choice = st.selectbox(
        "Choose ML Model",
        [
            "XGBoost",
            "Gradient Boosting", 
            "Random Forest",
            "Logistic Regression"
        ],
        help="Select the machine learning algorithm"
    )
    
    # Model performance info
    model_descriptions = {
        "XGBoost": "🏆 **Best Performance** - Industry standard (87-89% accuracy)",
        "Gradient Boosting": "🚀 **Excellent** - Sequential learning (86-88% accuracy)",
        "Random Forest": "🌲 **Very Good** - Robust ensemble (85-87% accuracy)",
        "Logistic Regression": "⚡ **Fast & Interpretable** - Baseline (85-86% accuracy)"
    }
    st.info(model_descriptions[model_choice])
    
    # Information section
    st.subheader("ℹ️ About Churn Prediction")
    st.info("""
    **Customer Churn** occurs when customers stop doing business with a company.
    
    **Why predict churn?**
    - Retain valuable customers
    - Reduce revenue loss
    - Target retention campaigns
    - Improve customer satisfaction
    
    **How it works:**
    1. Upload customer data
    2. Train ML model
    3. Predict churn probability
    4. Take preventive action
    """)
    
    st.markdown("---")
    st.caption("💡 Built for internship portfolio")

# ============================================================================
# MAIN CONTENT
# ============================================================================

if uploaded_file is not None:
    # Load data
    df = load_and_validate_data(uploaded_file)
    
    if df is not None:
        # Store in session state
        if 'df' not in st.session_state:
            st.session_state.df = df
        
        # Create tabs
        tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🤖 Model Training", "🔮 Predict New Customer"])
        
        # ====================================================================
        # TAB 1: DASHBOARD
        # ====================================================================
        with tab1:
            st.header("📊 Dataset Overview")
            
            # Dataset preview
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Customers", len(df))
            with col2:
                churn_count = df['Churn'].value_counts().get('Yes', 0)
                st.metric("Churned Customers", churn_count)
            with col3:
                churn_rate = (churn_count / len(df)) * 100
                st.metric("Churn Rate", f"{churn_rate:.1f}%")
            
            # Show sample data
            st.subheader("📋 Sample Data")
            st.dataframe(df.head(10), use_container_width=True)
            
            # Churn distribution
            st.subheader("📈 Churn Distribution")
            fig_dist = plot_churn_distribution(df)
            st.pyplot(fig_dist)
            
            # Data summary
            with st.expander("🔍 View Dataset Statistics"):
                st.write(df.describe())
        
        # ====================================================================
        # TAB 2: MODEL TRAINING
        # ====================================================================
        with tab2:
            st.header("🤖 Machine Learning Model")
            
            if st.button("🚀 Train Model", type="primary", use_container_width=True):
                with st.spinner("Training model... Please wait"):
                    # Preprocess data
                    X, y, feature_names, cat_cols, num_cols = preprocess_data(df)
                    
                    # Train-test split (80-20)
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42, stratify=y
                    )
                    
                    # Train model
                    model, scaler, accuracy, y_pred, y_pred_proba, roc_auc = train_model(
                        X_train, X_test, y_train, y_test, model_choice
                    )
                    
                    # Store in session state
                    st.session_state.model = model
                    st.session_state.scaler = scaler
                    st.session_state.feature_names = feature_names
                    st.session_state.cat_cols = cat_cols
                    st.session_state.num_cols = num_cols
                    st.session_state.X_train = X_train
                    
                    st.success("✅ Model trained successfully!")
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Model Accuracy", f"{accuracy*100:.2f}%")
                    with col2:
                        st.metric("ROC-AUC Score", f"{roc_auc:.3f}")
                    with col3:
                        st.metric("Model Type", model_choice)
                    
                    # Performance interpretation
                    if accuracy >= 0.87:
                        st.success("🎉 Outstanding! Production-ready model.")
                    elif accuracy >= 0.85:
                        st.success("✅ Excellent! Model exceeds industry standards.")
                    elif accuracy >= 0.80:
                        st.info("👍 Very good performance!")
                    else:
                        st.warning("⚠️ Consider trying a different model.")
                    
                    # Visualizations side by side
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("📊 Confusion Matrix")
                        fig_cm = plot_confusion_matrix(y_test, y_pred)
                        st.pyplot(fig_cm)
                    
                    with col2:
                        st.subheader("📈 ROC Curve")
                        fig_roc = plot_roc_curve(y_test, y_pred_proba)
                        st.pyplot(fig_roc)
                    
                    # Classification report
                    with st.expander("📋 Detailed Classification Report"):
                        report = classification_report(y_test, y_pred, 
                                                      target_names=['No Churn', 'Churn'])
                        st.text(report)
                        st.markdown("""
                        **Metrics Explained:**
                        - **Precision**: Of predicted churners, how many actually churned?
                        - **Recall**: Of actual churners, how many did we catch?
                        - **F1-Score**: Balance between precision and recall
                        """)
                    
                    # Feature importance (for tree-based models)
                    if model_choice in ["Random Forest", "Gradient Boosting", "XGBoost"]:
                        st.subheader("🎯 Feature Importance")
                        fig_imp = plot_feature_importance(model, feature_names, top_n=15)
                        if fig_imp:
                            st.pyplot(fig_imp)
                            st.caption("Top 15 features that most influence churn prediction")
            
            # Show training status
            if 'model' in st.session_state:
                st.info(f"✅ {model_choice} model is ready for predictions!")
            else:
                st.warning("⚠️ Please train the model first by clicking the button above.")
        
        # ====================================================================
        # TAB 3: PREDICTION
        # ====================================================================
        with tab3:
            st.header("🔮 Predict Customer Churn")
            
            if 'model' not in st.session_state:
                st.warning("⚠️ Please train a model first in the 'Model Training' tab!")
            else:
                st.info("👇 Enter customer information below to predict churn probability")
                
                # Define original features (only those in the uploaded dataset)
                # Exclude engineered features that are created during preprocessing
                original_categorical = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                                       'MultipleLines', 'InternetService', 'OnlineSecurity',
                                       'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                       'StreamingTV', 'StreamingMovies', 'Contract',
                                       'PaperlessBilling', 'PaymentMethod']
                
                original_numeric = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
                
                # Filter to only include columns that exist in the uploaded dataset
                available_categorical = [col for col in original_categorical if col in df.columns]
                available_numeric = [col for col in original_numeric if col in df.columns]
                
                # Create input form
                with st.form("prediction_form"):
                    st.subheader("Customer Information")
                    
                    # Create columns for better layout
                    col1, col2 = st.columns(2)
                    
                    input_data = {}
                    
                    # Categorical features (only from original dataset)
                    for i, col in enumerate(available_categorical):
                        unique_values = df[col].unique().tolist()
                        
                        with col1 if i % 2 == 0 else col2:
                            input_data[col] = st.selectbox(
                                f"{col.replace('_', ' ').title()}",
                                options=unique_values,
                                key=f"cat_{col}"
                            )
                    
                    # Numeric features (only from original dataset)
                    for i, col in enumerate(available_numeric):
                        # Convert to numeric and handle any non-numeric values
                        numeric_series = pd.to_numeric(df[col], errors='coerce')
                        
                        min_val = float(numeric_series.min())
                        max_val = float(numeric_series.max())
                        mean_val = float(numeric_series.mean())
                        
                        with col1 if i % 2 == 0 else col2:
                            input_data[col] = st.number_input(
                                f"{col.replace('_', ' ').title()}",
                                min_value=min_val,
                                max_value=max_val,
                                value=mean_val,
                                key=f"num_{col}"
                            )
                    
                    # Submit button (REQUIRED for forms)
                    st.markdown("<br>", unsafe_allow_html=True)
                    submitted = st.form_submit_button(
                        "🔮 Predict Churn", 
                        type="primary",
                        use_container_width=True
                    )
                
                # Make prediction when form is submitted
                if submitted:
                    # Create a single-row dataframe with the input
                    input_df = pd.DataFrame([input_data])
                    
                    # Add a dummy Churn column (required for preprocessing)
                    input_df['Churn'] = 'No'
                    
                    # Apply the SAME preprocessing as training
                    # This will create all the engineered features
                    try:
                        X_input, _, _, _, _ = preprocess_data(input_df)
                        
                        # Ensure all features match training data
                        for col in st.session_state.feature_names:
                            if col not in X_input.columns:
                                X_input[col] = 0
                        
                        # Reorder columns to match training data
                        X_input = X_input[st.session_state.feature_names]
                        
                        # Scale features using the saved scaler
                        input_scaled = st.session_state.scaler.transform(X_input)
                        
                        # Make prediction
                        prediction = st.session_state.model.predict(input_scaled)[0]
                        probability = st.session_state.model.predict_proba(input_scaled)[0][1]
                        
                        # Get risk level
                        risk_level, risk_color = get_risk_level(probability)
                        
                        # Display results
                        st.markdown("---")
                        st.subheader("📊 Prediction Results")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>Churn Prediction</h3>
                                <h2>{"❌ YES" if prediction == 1 else "✅ NO"}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>Churn Probability</h3>
                                <h2>{probability*100:.1f}%</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown(f"""
                            <div class="metric-card" style="background-color: {risk_color}20;">
                                <h3>Risk Level</h3>
                                <h2>{risk_level}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Recommendations
                        st.markdown("---")
                        st.subheader("💡 Recommended Actions")
                        
                        if prediction == 1:
                            st.error("""
                            **⚠️ High Churn Risk - Take Immediate Action!**
                            
                            Recommended retention strategies:
                            - 📞 **Contact customer immediately** - Understand their concerns
                            - 💰 **Offer special discount** - Provide loyalty incentives (10-20% off)
                            - 🎁 **Upgrade plan benefits** - Add premium services at no extra cost
                            - 👥 **Assign dedicated support** - Provide personalized assistance
                            - 📊 **Conduct satisfaction survey** - Identify pain points
                            - 🔒 **Lock-in promotion** - Encourage contract upgrade with benefits
                            """)
                        else:
                            st.success("""
                            **✅ Low Churn Risk - Customer Satisfied**
                            
                            Recommended actions:
                            - 🌟 **Maintain quality service** - Keep customer happy
                            - 📧 **Regular engagement** - Send helpful communications monthly
                            - 🎯 **Upsell opportunities** - Offer relevant premium features
                            - 💬 **Collect feedback** - Continuous improvement
                            - 🎁 **Loyalty rewards** - Recognize long-term customers
                            """)
                        
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")
                        st.info("Please ensure all fields are filled correctly and try again.")

else:
    # Welcome screen when no file is uploaded
    st.info("""
    ### 👋 Welcome to Telco Customer Churn Prediction System!
    
    This application helps telecom companies predict which customers are likely to churn (stop using service).
    
    **📝 To get started:**
    1. Upload the Telco Customer Churn dataset (CSV) using the sidebar
    2. Explore the data in the Dashboard tab
    3. Train a machine learning model
    4. Predict churn for new customers
    
    **📥 Dataset:**
    - Download from: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
    - Required column: `Churn` (Yes/No)
    
    **🎯 Business Value:**
    - Identify at-risk customers early
    - Reduce customer acquisition costs
    - Improve customer retention rates
    - Increase overall profitability
    """)
    
    # Sample data preview
    st.markdown("---")
    st.subheader("📋 Expected Dataset Format")
    sample_data = {
        'customerID': ['7590-VHVEG', '5575-GNVDE'],
        'gender': ['Female', 'Male'],
        'SeniorCitizen': [0, 0],
        'Partner': ['Yes', 'No'],
        'tenure': [1, 34],
        'MonthlyCharges': [29.85, 56.95],
        'TotalCharges': [29.85, 1889.5],
        'Churn': ['No', 'No']
    }
    st.dataframe(pd.DataFrame(sample_data), use_container_width=True)

# Footer
st.markdown("---")
st.caption("""
📌 **Project Info:** Telco Customer Churn Prediction System  
🛠️ **Tech Stack:** Python, Scikit-learn, Streamlit, Pandas  
""")

