
# Multi-Output Prediction Model Web Application

This is a **Streamlit**-based web application that uses a pre-trained **Random Forest Regressor** machine learning model to make predictions based on user-uploaded Excel data. The application calculates useful statistical insights such as frequency counts and percentages and allows users to download the results in an Excel file containing predictions, frequency counts, and formatted output.

## Features

- Upload Excel files for predictions.
- Make multi-output predictions using a pre-trained Random Forest Regressor model.
- Calculate frequency counts and percentages for the predicted output values.
- Download results as an Excel file with predictions, frequency counts, and formatted results.

## Prerequisites

Before running the application, ensure the following software is installed:

- Python 3.7+
- The following Python libraries:
  - `streamlit`
  - `pandas`
  - `numpy`
  - `joblib`
  - `xlsxwriter`

You can install these dependencies with the following command:

```bash
pip install streamlit pandas numpy joblib xlsxwriter
```

## Random Forest Regressor Model

### How Random Forest Works

The **Random Forest Regressor** is an ensemble learning algorithm that combines predictions from multiple decision trees. Rather than relying on a single decision tree, Random Forest generates multiple trees (the "forest") using different random subsets of the training data and features, and then averages their predictions for regression tasks.

### Algorithm Steps

1. **Bootstrap Sampling**: Multiple random samples with replacement are drawn from the training data to create subsets.
2. **Tree Construction**: A decision tree is built for each subset by recursively splitting the data based on the best feature split.
3. **Random Feature Selection**: At each node in the tree, a random subset of features is selected to determine the split, which helps decorrelate the trees.
4. **Aggregation**: For regression tasks, the predictions from all decision trees are averaged to form the final output.

### Mathematical Formula

Let there be `n` decision trees, each producing a prediction \( \hat{y}_i \) for an input \( X \). The final prediction of the Random Forest is:

\[
\hat{y} = rac{1}{n} \sum_{i=1}^{n} \hat{y}_i
\]

Where:
- \( \hat{y} \) is the final predicted value.
- \( \hat{y}_i \) is the predicted value from the \( i \)-th decision tree.
- \( n \) is the number of decision trees.

### Hyperparameters Used

The following hyperparameters were tuned using **GridSearchCV**:
- `n_estimators`: The number of decision trees in the forest (tested values: 100, 200, 300).
- `max_depth`: The maximum depth of the trees (tested values: `None`, 5, 10, 15).
- `random_state`: A fixed seed to ensure reproducibility.

### Model Training

The model was trained using a dataset with the following input columns:

- `Technology Acceptance`
- `Level of use of AI based tools`
- `Technology based Tutoring System`
- `Organisational Performance`
- `Student's Performance`

The output columns represent ordinal values that classify data into categories such as `Very Low`, `Low`, `Moderate`, `High`, and `Very High`.

## How to Run the Application

1. **Clone the Repository**:
   Clone the repository to your local machine:

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Add the Pre-trained Model**:
   Ensure the `random_forest_model.joblib` file (containing the pre-trained model) is placed in the project directory.

3. **Run the Streamlit Application**:
   Start the Streamlit server by running the following command:

   ```bash
   streamlit run main.py
   ```

4. **Upload Excel File**:
   - Open the URL provided by Streamlit (typically http://localhost:8501).
   - Upload an Excel file containing the necessary input columns.

5. **Download Predictions and Results**:
   - After predictions are made and frequency counts are calculated, a download link will be available.
   - The downloadable Excel file will contain the following sheets:
     - **Predictions**: Input data with predictions.
     - **Frequency_Counts**: Frequency of each category.
     - **Result**: A formatted sheet summarizing the frequency counts and percentages.

## Folder Structure

```bash
.
├── main.py                     # Main Streamlit application
├── random_forest_model.joblib   # Pre-trained Random Forest Regressor model (not included)
├── README.md                    # This README file
```

## Input and Output Structure

### Required Input Columns:

The application expects the following columns in the uploaded Excel file:

- `Technology Acceptance`
- `Level of use of AI based tools`
- `Technology based Tutoring System`
- `Organisational Performance`
- `Student's Performance`

### Output:

The application provides the following outputs:

1. **Predictions**: The model's predicted values for each row of input data.
2. **Frequency Counts**: The frequency of predicted categories (`Very Low`, `Low`, `Moderate`, `High`, `Very High`).
3. **Formatted Results**: A table summarizing the frequency counts and percentages.

## Known Issues

- Input columns must match exactly in name and format.
- The model currently handles missing values by filling them with the column mean.

## License

This project is licensed under the Runita.
