# Credit Card Fraud Detection System

## Overview
This project implements a machine learning-based fraud detection system for credit card transactions. The system uses a Logistic Regression model to classify transactions as either legitimate or fraudulent based on various transaction features.

## Dataset
The model was trained on a credit card transaction dataset that includes the following key features:
- **Transaction Amount**: The monetary value of the transaction
- **Time**: The time when the transaction occurred
- **V1-V28**: Principal components derived from the original transaction features (PCA-transformed for privacy and security)
- **Class**: Binary target variable (0 for legitimate transactions, 1 for fraudulent transactions)

The dataset has been preprocessed using Principal Component Analysis (PCA) to protect sensitive information while maintaining the essential patterns needed for fraud detection. The original features have been transformed into 28 principal components (V1-V28), with only V1 and V2 being exposed in the interface for demonstration purposes.

## Model Architecture
- **Algorithm**: Logistic Regression
- **Preprocessing**: StandardScaler for feature normalization
- **Input Features**: 30-dimensional feature vector (including amount, time, and PCA components)
- **Output**: Binary classification (0: Legitimate, 1: Fraudulent) with probability scores

## Model Selection and Performance
During the development of this fraud detection system, several machine learning models were evaluated:

1. **Logistic Regression (Selected Model)**
   - Advantages:
     - Fast training and prediction times
     - Good interpretability
     - Works well with imbalanced datasets
     - Provides probability scores
   - Performance Metrics:
     - High precision in fraud detection
     - Low false positive rate
     - Balanced accuracy for both classes

2. **Random Forest**
   - Advantages:
     - Handles non-linear relationships
     - Good at handling outliers
   - Limitations:
     - Slower prediction times
     - More complex to interpret
     - Higher computational requirements

3. **Support Vector Machine (SVM)**
   - Advantages:
     - Effective in high-dimensional spaces
     - Good for non-linear classification
   - Limitations:
     - Slower training and prediction
     - Sensitive to feature scaling
     - Higher memory requirements

4. **Neural Network**
   - Advantages:
     - Can capture complex patterns
     - Good at learning non-linear relationships
   - Limitations:
     - Requires more data
     - Longer training time
     - More complex to tune and maintain

### Why Logistic Regression?
The Logistic Regression model was selected as the final model because:
1. **Performance**: It achieved the best balance between precision and recall for fraud detection
2. **Speed**: Provides real-time predictions crucial for fraud detection
3. **Interpretability**: Easy to understand and explain predictions
4. **Resource Efficiency**: Low computational requirements
5. **Probability Scores**: Provides reliable probability estimates for risk assessment

The model was trained on a balanced dataset using techniques to handle class imbalance, and it was validated using cross-validation to ensure robust performance.

## Model Performance Visualizations

Below are some infographics to help visualize the performance of the models and the selected Logistic Regression model:

### 1. Model Performance Comparison
![Model Comparison]
*Comparison of accuracy, precision, and recall for different models. Logistic Regression achieved the best balance for this task.*
![model_comparison](https://github.com/user-attachments/assets/ee66852b-85b2-4928-bb2b-bcd625842e82)


### 2. Confusion Matrix
![Confusion Matrix]
*Shows the number of correct and incorrect predictions for legitimate and fraudulent transactions.*
![confusion_matrix](https://github.com/user-attachments/assets/665b2d2f-b475-47fb-859d-6b319c5c8128)


### 3. ROC Curve
![ROC Curve](images/roc_curve.png)
*The ROC curve demonstrates the trade-off between true positive rate and false positive rate. The area under the curve (AUC) indicates strong model performance.*
![roc_curve](https://github.com/user-attachments/assets/a4876f11-340c-4cba-88d9-2ab036feac32)


### 4. Feature Importance
![Feature Importance]
*Relative importance of the top features used by the model for prediction.*
![feature_importance](https://github.com/user-attachments/assets/6bfe2883-5fad-4468-9fda-1e3d41a9d998)


### 5. Precision-Recall Curve
![Precision-Recall Curve]
*Shows the trade-off between precision and recall, which is especially important for imbalanced datasets like fraud detection.*
![precision_recall_curve](https://github.com/user-attachments/assets/a9c4bc26-e1bd-4137-b7b0-f25831bc8772)


## Technical Stack
- Python 3.x
- Streamlit (v1.32.0) for the web interface
- scikit-learn (v1.4.0) for machine learning
- pandas (v2.2.0) for data manipulation
- numpy (v1.26.4) for numerical operations

## Project Structure
```
├── app.py              # Streamlit web application
├── lr_model.pkl        # Trained Logistic Regression model
├── scaler.pkl          # StandardScaler for feature normalization
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Getting Started

### Prerequisites
- Python 3.x
- pip (Python package installer)

### Installation
1. Clone the repository
2. Install the required packages:
```bash
pip install -r requirements.txt
```

### Running the Application
To start the fraud detection system:
```bash
streamlit run app.py
```

## Usage
1. Enter the transaction details:
   - Transaction Amount
   - Time of Transaction
   - V1 (First Principal Component)
   - V2 (Second Principal Component)
2. Click "Predict Fraud" to get the prediction
3. View the results:
   - Transaction classification (Legitimate/Fraudulent)
   - Confidence score of the prediction

## Model Performance
The model provides:
- Binary classification (Legitimate/Fraudulent)
- Probability scores for each prediction
- Real-time predictions with confidence levels

## Security Note
The model uses PCA-transformed features to ensure the privacy and security of the original transaction data. The actual feature values are not exposed, maintaining the confidentiality of the transaction information.

## Limitations
- The model's performance depends on the quality and representativeness of the training data
- The current implementation only exposes V1 and V2 components for demonstration
- The model should be regularly retrained with new data to maintain its effectiveness

## Future Improvements
- Add more features to the interface
- Implement model retraining pipeline
- Add real-time transaction monitoring
- Include model performance metrics
- Add support for batch predictions

## License
This project is licensed under the MIT License - see the LICENSE file for details. 
