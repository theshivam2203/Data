import streamlit as st
import pandas as pd
import joblib
from io import BytesIO

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

# Function to save the result to Excel, updating existing sheets or adding new ones
def to_excel_with_updates(original_sheets, result_df, frequency_combined_df):
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
        
    processed_data = output.getvalue()
    return processed_data

# Define the model filename
model_filename = 'random_forest_model.joblib'

# Load the model
model = load_model(model_filename)

# Streamlit app interface
st.title("Multi-Output Prediction Model")

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

            # Display the result
            st.header("Predictions")
            st.write(result_df)

            # Display frequency counts
            st.header("Frequency Counts")
            st.write(frequency_combined_df)

            # Option to download the result as an Excel file
            st.download_button(
                label="Download Predictions and Frequency Counts as Excel",
                data=to_excel_with_updates(original_sheets, result_df, frequency_combined_df),
                file_name='predictions_and_frequency_counts.xlsx',
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
            )
        except Exception as e:
            st.error(f"An error occurred while making predictions: {e}")
    else:
        st.error(f"The uploaded file does not contain the required columns: {', '.join(missing_columns)}.")
else:
    st.info("Please upload an Excel file.")


# import streamlit as st
# import pandas as pd
# import joblib
# from io import BytesIO

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
#     if all(column in df.columns for column in input_columns):
#         input_df = df[input_columns]

#         # Make predictions
#         predictions = make_predictions(model, input_df)

#         # Define the output columns
#         output_columns = [
#             'Technology_Acceptance_Range',
#             'Level_of_use_of_AI_based_tools_Range',
#             'Technology_based_Tutoring_System_Range',
#             'Organisational_Performance_Range',
#             'Student\'s_Performance_Range'
#         ]

#         # Create a DataFrame for the predictions
#         prediction_df = pd.DataFrame(predictions, columns=output_columns)

#         # Concatenate the input data with the predictions
#         result_df = pd.concat([df, prediction_df], axis=1)

#         # Calculate frequency counts and percentages
#         frequency_combined_df = calculate_frequency_counts_and_percentages(result_df, output_columns)

#         # Display the result
#         st.header("Predictions")
#         st.write(result_df)

#         # Display frequency counts
#         st.header("Frequency Counts")
#         st.write(frequency_combined_df)

#         # Option to download the result as an Excel file
#         st.download_button(
#             label="Download Predictions and Frequency Counts as Excel",
#             data=to_excel_with_updates(original_sheets, result_df, frequency_combined_df),
#             file_name='predictions_and_frequency_counts.xlsx',
#             mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
#         )
#     else:
#         st.error("The uploaded file does not contain the required columns.")
# else:
#     st.info("Please upload an Excel file.")

