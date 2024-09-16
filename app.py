import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
import numpy as np

# Function to load the model
def load_model(model_filename):
    """Load a saved model from disk."""
    return joblib.load(model_filename)

# Function to make predictions
def make_predictions(model, input_df):
    """Make predictions using the loaded model and input dataframe."""
    predictions = model.predict(input_df)
    rounded_predictions = predictions.round().astype(int)
    return rounded_predictions

# Function to calculate frequency counts and percentages
def calculate_frequency_counts_and_percentages(df, output_columns):
    """Calculate frequency counts and percentages for each output column."""
    all_values = range(df[output_columns].min().min(), df[output_columns].max().max() + 1)
    frequency_counts_dict = {}
    total_counts = len(df)

    for column in output_columns:
        frequency_counts = df[column].value_counts().reindex(all_values, fill_value=0).reset_index()
        frequency_counts.columns = ['Value', f'{column}_Frequency']
        frequency_counts[f'{column}_Percentage'] = (frequency_counts[f'{column}_Frequency'] / total_counts) * 100
        frequency_counts_dict[column] = frequency_counts

    # Combine all frequency counts into a single DataFrame
    frequency_combined_df = pd.concat(frequency_counts_dict.values(), axis=1)
    frequency_combined_df = frequency_combined_df.loc[:, ~frequency_combined_df.columns.duplicated()]
    return frequency_combined_df


# Function to create the formatted 'result' DataFrame dynamically based on frequency counts
def create_formatted_result_sheet(frequency_combined_df):
    """Create a properly formatted result sheet based on frequency data."""
    categories = [
        'Level of Technology Acceptance',
        'Level of use of AI based tools',
        'Level of Technology based Tutoring System',
        'Level of Organisational Performance',
        'Level of Student\'s Performance'
    ]

    result_data = {'Category': [], 'Range': [], 'Frequency': [], 'Percentage': []}
    
    # Extract data from frequency_combined_df for each category and format it
    for idx, category in enumerate(categories):
        result_data['Category'].extend([category] + [''] * 5)  # Category name and blank rows
        result_data['Range'].extend(['Very Low', 'Low', 'Moderate', 'High', 'Very high', 'Total'])

        freq_col = frequency_combined_df.columns[idx * 2 + 1]  # Frequency column
        perc_col = frequency_combined_df.columns[idx * 2 + 2]  # Percentage column
        
        freq_values = frequency_combined_df[freq_col].values
        perc_values = frequency_combined_df[perc_col].values

        # Append the frequencies and percentages
        result_data['Frequency'].extend(list(freq_values))

        # Calculate and append totals
        total_freq = sum(freq_values)
        total_perc = 100  # Since percentages should sum to 100
        result_data['Frequency'].append(total_freq)
        result_data['Percentage'].extend(list(perc_values))
        result_data['Percentage'].append(total_perc)

    return pd.DataFrame(result_data)


# Function to save the result to Excel, updating existing sheets or adding new ones
def to_excel_with_updates(original_sheets, result_df, frequency_combined_df, formatted_result_df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        for sheet_name, sheet_df in original_sheets.items():
            if sheet_name == 'Predictions':
                sheet_df = result_df  # Update the 'Predictions' sheet
            elif sheet_name == 'Frequency_Counts':
                sheet_df = frequency_combined_df  # Update the 'Frequency_Counts' sheet
            sheet_df.to_excel(writer, index=False, sheet_name=sheet_name)
        
        # Add new sheets if they don't exist
        if 'Predictions' not in original_sheets:
            result_df.to_excel(writer, index=False, sheet_name='Predictions')
        if 'Frequency_Counts' not in original_sheets:
            frequency_combined_df.to_excel(writer, index=False, sheet_name='Frequency_Counts')

        # Add the formatted result sheet
        formatted_result_df.to_excel(writer, index=False, sheet_name='Result')

    processed_data = output.getvalue()
    return processed_data

# Define the model filename
model_filename = 'random_forest_model.joblib'

# Load the model
model = load_model(model_filename)

# Streamlit app interface
st.title("Comprehensive Data Analysis for Evaluating Technology Acceptance and Performance in AI-Based Tools")

# Display required input columns
st.write("The input data should contain the following columns:")
required_input_columns = [
    "- 'Technology Acceptance'",
    "- 'Level of use of AI based tools'",
    "- 'Technology based Tutoring System'",
    "- 'Organisational Performance'",
    "- 'Student's Performance'"
]
for column in required_input_columns:
    st.write(column)

# File uploader for Excel files
uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file is not None:
    # Read all sheets from the uploaded file
    original_sheets = pd.read_excel(uploaded_file, sheet_name=None)

    # Select the first sheet as default input sheet (or allow user to select)
    default_sheet_name = list(original_sheets.keys())[0]
    df = original_sheets[default_sheet_name]

    # Define the input columns (make sure they match your model's training data)
    input_columns = [
        'Technology Acceptance',
        'Level of use of AI based tools',
        'Technology based Tutoring System',
        'Organisational Performance',
        'Student\'s Performance'
    ]

    # Check if all required columns are present
    missing_columns = [column for column in input_columns if column not in df.columns]
    if not missing_columns:
        input_df = df[input_columns]

        # Handle missing values in the input data
        if input_df.isnull().values.any():
            st.warning("The input data contains missing values. Filling missing values with the column mean.")
            input_df = input_df.fillna(input_df.mean())

        # Make predictions
        try:
            predictions = make_predictions(model, input_df)

            # Define the output columns
            output_columns = [
                'Technology_Acceptance_Range',
                'Level_of_use_of_AI_based_tools_Range',
                'Technology_based_Tutoring_System_Range',
                'Organisational_Performance_Range',
                'Student\'s_Performance_Range'
            ]

            # Create a DataFrame for the predictions
            prediction_df = pd.DataFrame(predictions, columns=output_columns)

            # Concatenate the input data with the predictions
            result_df = pd.concat([df, prediction_df], axis=1)

            # Calculate frequency counts and percentages
            frequency_combined_df = calculate_frequency_counts_and_percentages(result_df, output_columns)

            # Create the formatted result sheet based on the frequency counts
            formatted_result_df = create_formatted_result_sheet(frequency_combined_df)

            # Option to download the result as an Excel file
            st.write("The predictions and frequency counts will be available for download.")
            st.download_button(
                label="Download Predictions, Frequency Counts, and Result as Excel",
                data=to_excel_with_updates(original_sheets, result_df, frequency_combined_df, formatted_result_df),
                file_name='predictions_and_result_sheet.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        except Exception as e:
            st.error(f"An error occurred while making predictions: {e}")
    else:
        st.error(f"The uploaded file does not contain the required columns: {', '.join(missing_columns)}.")
else:
    st.info("Please upload an Excel file.")























#old Working code


# import streamlit as st
# import pandas as pd
# import joblib
# from io import BytesIO
# import numpy as np

# # Function to load the model
# def load_model(model_filename):
#     """Load a saved model from disk."""
#     return joblib.load(model_filename)

# # Function to make predictions
# def make_predictions(model, input_df):
#     """Make predictions using the loaded model and input dataframe."""
#     predictions = model.predict(input_df)
#     rounded_predictions = predictions.round().astype(int)
#     return rounded_predictions

# # Function to calculate frequency counts and percentages
# def calculate_frequency_counts_and_percentages(df, output_columns):
#     """Calculate frequency counts and percentages for each output column."""
#     all_values = range(df[output_columns].min().min(), df[output_columns].max().max() + 1)
#     frequency_counts_dict = {}
#     total_counts = len(df)

#     for column in output_columns:
#         frequency_counts = df[column].value_counts().reindex(all_values, fill_value=0).reset_index()
#         frequency_counts.columns = ['Value', f'{column}_Frequency']
#         frequency_counts[f'{column}_Percentage'] = (frequency_counts[f'{column}_Frequency'] / total_counts) * 100
#         frequency_counts_dict[column] = frequency_counts

#     # Combine all frequency counts into a single DataFrame
#     frequency_combined_df = pd.concat(frequency_counts_dict.values(), axis=1)
#     frequency_combined_df = frequency_combined_df.loc[:, ~frequency_combined_df.columns.duplicated()]
#     return frequency_combined_df

# # Function to save the result to Excel, updating existing sheets or adding new ones
# def to_excel_with_updates(original_sheets, result_df, frequency_combined_df):
#     output = BytesIO()
#     with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
#         for sheet_name, sheet_df in original_sheets.items():
#             if sheet_name == 'Predictions':
#                 sheet_df = result_df  # Update the 'Predictions' sheet
#             elif sheet_name == 'Frequency_Counts':
#                 sheet_df = frequency_combined_df  # Update the 'Frequency_Counts' sheet
#             sheet_df.to_excel(writer, index=False, sheet_name=sheet_name)
        
#         # Add new sheets if they don't exist
#         if 'Predictions' not in original_sheets:
#             result_df.to_excel(writer, index=False, sheet_name='Predictions')
#         if 'Frequency_Counts' not in original_sheets:
#             frequency_combined_df.to_excel(writer, index=False, sheet_name='Frequency_Counts')
        
#     processed_data = output.getvalue()
#     return processed_data

# # Define the model filename
# model_filename = 'random_forest_model.joblib'

# # Load the model
# model = load_model(model_filename)

# # Streamlit app interface
# st.title("Multi-Output Prediction Model")

# # Display required input columns
# st.write("The input data should contain the following columns:")
# required_input_columns = [
#     "- 'Technology Acceptance'",
#     "- 'Level of use of AI based tools'",
#     "- 'Technology based Tutoring System'",
#     "- 'Organisational Performance'",
#     "- 'Student's Performance'"
# ]
# for column in required_input_columns:
#     st.write(column)

# # File uploader for Excel files
# uploaded_file = st.file_uploader("Upload an Excel file", type=["xlsx"])

# if uploaded_file is not None:
#     # Read all sheets from the uploaded file
#     original_sheets = pd.read_excel(uploaded_file, sheet_name=None)

#     # Select the first sheet as default input sheet (or allow user to select)
#     default_sheet_name = list(original_sheets.keys())[0]
#     df = original_sheets[default_sheet_name]

#     # Define the input columns (make sure they match your model's training data)
#     input_columns = [
#         'Technology Acceptance',
#         'Level of use of AI based tools',
#         'Technology based Tutoring System',
#         'Organisational Performance',
#         'Student\'s Performance'
#     ]

#     # Check if all required columns are present
#     missing_columns = [column for column in input_columns if column not in df.columns]
#     if not missing_columns:
#         input_df = df[input_columns]

#         # Handle missing values in the input data
#         if input_df.isnull().values.any():
#             st.warning("The input data contains missing values. Filling missing values with the column mean.")
#             input_df = input_df.fillna(input_df.mean())

#         # Make predictions
#         try:
#             predictions = make_predictions(model, input_df)

#             # Define the output columns
#             output_columns = [
#                 'Technology_Acceptance_Range',
#                 'Level_of_use_of_AI_based_tools_Range',
#                 'Technology_based_Tutoring_System_Range',
#                 'Organisational_Performance_Range',
#                 'Student\'s_Performance_Range'
#             ]

#             # Create a DataFrame for the predictions
#             prediction_df = pd.DataFrame(predictions, columns=output_columns)

#             # Concatenate the input data with the predictions
#             result_df = pd.concat([df, prediction_df], axis=1)

#             # Calculate frequency counts and percentages
#             frequency_combined_df = calculate_frequency_counts_and_percentages(result_df, output_columns)

#             # Option to download the result as an Excel file
#             st.write("The predictions and frequency counts will be available for download.")
#             st.download_button(
#                 label="Download Predictions and Frequency Counts as Excel",
#                 data=to_excel_with_updates(original_sheets, result_df, frequency_combined_df),
#                 file_name='predictions_and_frequency_counts.xlsx',
#                 mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
#             )
#         except Exception as e:
#             st.error(f"An error occurred while making predictions: {e}")
#     else:
#         st.error(f"The uploaded file does not contain the required columns: {', '.join(missing_columns)}.")
# else:
#     st.info("Please upload an Excel file.")
