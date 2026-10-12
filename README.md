# Diabetes Risk Prediction using Machine Learning and Neural Networks

This project implements and evaluates multiple machine learning and deep learning models to predict diabetes based on clinical and demographic patient data. By preprocessing real-world data, performing Exploratory Data Analysis (EDA), and addressing class imbalances, we build robust diagnostic classifiers.

## 🚀 Project Overview
- **Objective:** Build predictive models to determine whether a patient is diabetic based on features like age, BMI, HbA1c levels, and blood glucose.
- **Tech Stack:** Python, TensorFlow/Keras, Scikit-Learn, Pandas, NumPy, Seaborn, Matplotlib.
- **Dataset Characteristics:** 100,000 records with 9 features (highly imbalanced: ~92.4% non-diabetic, ~7.6% diabetic).

## 📊 Methodology & Workflow

### 1. Data Preprocessing & Pipeline
- **Imputation:** Median imputation for continuous variables (`age`, `bmi`, `HbA1c_level`, `blood_glucose_level`) and mode imputation for binary/categorical features to handle any missing values cleanly.
- **Feature Scaling:** Applied standard scaling (`StandardScaler`) to continuous variables to optimize gradient convergence for the neural network.
- **Categorical Encoding:** Leveraged `LabelEncoder` for categorical variables (`gender`, `smoking_history`).

### 2. Exploratory Data Analysis (EDA)
- Generated correlation heatmaps to explore linear relationships.
- Analyzed class imbalance to map skewed targets.
- Visualized distributions using histograms, boxplots, and density curves to understand outlier distribution (particularly in `BMI`).

### 3. Model Architecture & Training
We compared three distinct machine learning algorithms trained on a stratified 70/30 train-test split:
- **Deep Neural Network (Sequential API):** A 128-neuron hidden layer using ReLU activation, with a Softmax output layer. Trained with `Adam` optimizer and `SparseCategoricalCrossentropy` loss.
- **Logistic Regression:** Regularized linear classification baseline (`max_iter=1000`).
- **K-Nearest Neighbors (KNN):** Distance-weighted classification configured for $k=5$.

---

## 📈 Model Performance Summary

| Model | Accuracy | Precision | Recall | AUC |
| :--- | :---: | :---: | :---: | :---: |
| **Neural Network** | **96.27%** | *0.86* | *0.63* | *0.96* |
| **Logistic Regression** | **95.35%** | *0.83* | *0.49* | *0.95* |
| **KNN Classifier** | **95.27%** | *0.78* | *0.52* | *0.91* |

### Key Takeaways:
- **Neural Network** achieved the highest overall testing accuracy (**96.27%**) and the best area under the ROC curve (**AUC = 0.96**).
- Due to the dataset's **imbalanced nature**, Precision and Recall metrics were crucial. The Neural Network performed best at identifying diabetic patients (Recall: 0.63, Precision: 0.86) compared to traditional ML models.

---

## 🛠️ How to Run Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```
2. Install required dependencies:
   ```bash
   pip install pandas numpy scikit-learn tensorflow matplotlib seaborn
   ```
3. Run the notebook or script to train and evaluate the models.
```
