# Multi-Output Prediction Model Web Application

This is a **Streamlit**-based web application that loads a pre-trained machine learning model (Random Forest Regressor), takes an Excel file as input, makes predictions, and provides useful statistical insights like frequency counts and percentages. The application allows users to download the results in an Excel file containing the predictions, frequency counts, and formatted output.

## Features

- Upload Excel files containing data for predictions.
- Make multi-output predictions using a pre-trained Random Forest Regressor model.
- Calculate and display frequency counts and percentages for the predicted output.
- Download results in an Excel file with updated or new sheets containing predictions, frequency counts, and formatted results.

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.7+
- The following Python libraries:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `joblib`
  - `xlsxwriter`
  
  You can install these dependencies by running:
  
  ```bash
  pip install streamlit pandas numpy joblib xlsxwriter



Here's an extended README.md with details about the Random Forest model, its workings, algorithm, and mathematical formulas:

markdown
Copy code
# Multi-Output Prediction Model Web Application

This is a **Streamlit**-based web application that loads a pre-trained machine learning model (Random Forest Regressor), takes an Excel file as input, makes predictions, and provides useful statistical insights like frequency counts and percentages. The application allows users to download the results in an Excel file containing the predictions, frequency counts, and formatted output.

## Features

- Upload Excel files containing data for predictions.
- Make multi-output predictions using a pre-trained Random Forest Regressor model.
- Calculate and display frequency counts and percentages for the predicted output.
- Download results in an Excel file with updated or new sheets containing predictions, frequency counts, and formatted results.

## Prerequisites

Before running the application, ensure you have the following installed:

- Python 3.7+
- The following Python libraries:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `joblib`
  - `xlsxwriter`
  
  You can install these dependencies by running:
  
  ```bash
  pip install streamlit pandas numpy joblib xlsxwriter
Model Overview: Random Forest Regressor
How Random Forest Works
The Random Forest Regressor is an ensemble learning algorithm that combines multiple decision trees to improve the accuracy of predictions. Instead of relying on a single decision tree, a Random Forest creates multiple decision trees (hence the "forest") and aggregates their predictions.

Algorithm Steps
Bootstrap Sampling: The Random Forest algorithm takes multiple random samples (with replacement) from the training dataset to create different subsets of data for each decision tree.
Decision Tree Building: For each subset, a decision tree is built by recursively splitting the data based on feature values to minimize the error. The decision trees in a Random Forest are typically deep trees, meaning they are grown without pruning.
Random Feature Selection: At each node in the decision tree, a random subset of features is selected, and the best split is chosen from this subset. This randomness helps make the trees less correlated and improves the ensemble's predictive performance.
Voting/Averaging: For classification tasks, each decision tree "votes" for a class, and the class with the most votes is selected. For regression tasks (like in this case), the predictions from all decision trees are averaged to produce the final result.
Mathematical Formula
Let’s assume there are n decision trees in the forest, and each tree produces a prediction 
𝑦
^
𝑖
y
^
​
  
i
​
  for an input 
𝑋
X. The final prediction of the Random Forest is the average of all individual tree predictions:

𝑦
^
=
1
𝑛
∑
𝑖
=
1
𝑛
𝑦
^
𝑖
y
^
​
 = 
n
1
​
  
i=1
∑
n
​
  
y
^
​
  
i
​
 
Where:

𝑦
^
y
^
​
  is the final predicted value (regression output).
𝑦
^
𝑖
y
^
​
  
i
​
  is the predicted value from the 
𝑖
i-th decision tree.
𝑛
n is the total number of decision trees in the forest.
Advantages of Random Forest
Reduction in Overfitting: Random Forest reduces overfitting compared to a single decision tree because it combines the predictions of multiple trees.
Handles Large Datasets: It is robust for large datasets and can handle a large number of input features.
Feature Importance: Random Forest models provide insights into which features are the most important in making predictions.
Model Hyperparameters
The following hyperparameters were used when training the Random Forest Regressor:

n_estimators: The number of trees in the forest (typically 100, 200, or 300).
max_depth: The maximum depth of each tree (which controls overfitting).
random_state: A fixed seed for reproducibility of results.
In this application, GridSearchCV was used to tune the following hyperparameters:

n_estimators: Number of decision trees (tested values: 100, 200, 300).
max_depth: Maximum depth of each tree (tested values: None, 5, 10, 15).
How the Random Forest Model Was Trained
The model was trained using a dataset with the following input columns:

Technology Acceptance
Level of use of AI based tools
Technology based Tutoring System
Organisational Performance
Student's Performance
The goal was to predict ordinal values that represent categories such as Very Low, Low, Moderate, High, and Very High based on the input data.

The training process involved:

Data Preparation: Preprocessing the input features and output targets.
Hyperparameter Tuning: Using GridSearchCV to find the optimal values for the hyperparameters.
Model Training: Training the Random Forest model on 80% of the dataset.
Model Evaluation: Testing the model on the remaining 20% of the data to evaluate its performance using metrics like R² (R-squared), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
How to Run the Application
Clone the Repository: Clone this repository to your local machine using the following command:

bash
Copy code
git clone <repository-url>
cd <repository-directory>
Add the Model: Ensure you have the random_forest_model.joblib file in the project directory. This file should contain the pre-trained Random Forest Regressor model.

Run the Streamlit Application: Start the Streamlit server by running the following command:

bash
Copy code
streamlit run main.py
Upload Excel File:

Navigate to the URL provided by Streamlit after launching the application (usually http://localhost:8501).
Upload an Excel file containing data for the required input columns.
Download Predictions and Results:

After the predictions are made and the frequency counts are calculated, the application will provide a download link for the results.
The downloadable Excel file will contain the following sheets:
Predictions: The input data combined with the predictions from the model.
Frequency_Counts: A table showing the frequency of each category in the predictions.
Result: A formatted sheet showing frequency counts and percentages by category.
Folder Structure
bash
Copy code
.
├── main.py                     # Main Streamlit application
├── random_forest_model.joblib   # Pre-trained Random Forest Regressor model (not included)
├── README.md                    # This README file
Input and Output Structure
Required Input Columns:
The application expects an Excel file with the following columns:

Technology Acceptance
Level of use of AI based tools
Technology based Tutoring System
Organisational Performance
Student's Performance
Output:
The output consists of:

Predictions: Predicted values for each input row.
Frequency Counts: Frequency of each category (Very Low, Low, Moderate, High, Very High) for the predicted values.
Formatted Results: A user-friendly table summarizing the frequency counts and percentages by category.

Known Issues
The model requires input columns to match exactly in name and format.
Currently, the model handles missing values by filling them with the column mean.
License
This project is licensed under the Runita.
