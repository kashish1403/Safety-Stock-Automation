import pandas as pd
import numpy as np
import re
import warnings


# Function to rearrange columns
def rearrange_columns(input_file):
    # Read the Excel file
    df = pd.read_excel(input_file)
    # Extract date columns in "YYYY/MM" format
    date_columns = [col for col in df.columns if isinstance(col, str) and '/' in col and len(col) == 7]
    if not date_columns:
        print("No date columns found in 'YYYY/MM' format in the DataFrame.")
        return df  # You may choose a different behavior based on your requirements
    # Filter out invalid date columns
    valid_date_columns = [col for col in date_columns if col.split('/')[0].isdigit() and col.split('/')[1].isdigit()]
    if not valid_date_columns:
        print("No valid date columns found.")
        return df  # You may choose a different behavior based on your requirements
    # Identify the column with the highest date
    latest_date_column = max(valid_date_columns, key=lambda x: (int(x.split('/')[0]), int(x.split('/')[1])))
    # Sort valid date columns based on the year and month components
    sorted_columns = sorted(valid_date_columns, key=lambda x: (int(x.split('/')[0]), int(x.split('/')[1])))
    # Shift the latest date column to the last
    sorted_columns.remove(latest_date_column)
    sorted_columns.append(latest_date_column)
    # Create a new DataFrame with sorted and shifted date columns
    df_combined = df[sorted_columns]
    # Combine with non-date columns
    non_date_columns = [col for col in df.columns if col not in valid_date_columns]
    df_combined = pd.concat([df[non_date_columns], df_combined], axis=1)
    return df_combined

def replace_negatives_and_nan_with_zero(df):
    warnings.filterwarnings("ignore")
    # Replace negative values with zero
    df = df.applymap(lambda x: 0 if (isinstance(x, (int, float)) and x < 0) else x)
    # Replace NaN values with zero
    df = df.fillna(0)

    return df
# Function to calculate consumption using simple moving average
def calculate_consumption(numbers, window_size):
    numbers = np.nan_to_num(numbers, nan=0.0)  # Replace NaN with zeros
    moving_avg = np.convolve(numbers, np.ones(window_size) / window_size, mode='valid')
    return round(moving_avg[-1])


def main():
    # if len(sys.argv) != 2:
    #     print("Usage: python file.py <input_file_path>")
    #     sys.exit(1)
    #
    # input_file = sys.argv[1]
    # Read and rearrange columns
    # input_file = "C:/Users/Admin/Downloads/input_file.xlsx"
    input_file = "C:/Users/Admin/Downloads/input_file_new.xlsx"
    # if not (os.path.isfile(input_file)):
    #     print("Error: One or both of the specified files do not exist.")
    #     sys.exit(1)
    df = rearrange_columns(input_file)
    df = replace_negatives_and_nan_with_zero(df)
    # Find quartiles and replace values
    consumption_cols = [col for col in df.columns if re.match(r'\d{4}/\d{2}', col)]
    def calculate_quartiles(row):

        q1 = row[consumption_cols].quantile(0.1)
        q3 = row[consumption_cols].quantile(0.9)
        return q1, q3

    df[['Q1', 'Q3']] = df.apply(calculate_quartiles, axis=1, result_type='expand')

    def replace_with_quartiles(row):
        for col in consumption_cols:
            if row[col] < row['Q1']:
                row[col] = row['Q1']
            elif row[col] > row['Q3']:
                row[col] = row['Q3']
        return row

    result_df = pd.DataFrame()
    for material_id, group in df.groupby('Material'):
        group = group.apply(replace_with_quartiles, axis=1)
        result_df = pd.concat([result_df, group])

    # Predict consumption using outlier treated columns

    previous_months = int(input("Enter the number of previous months to consider for calculation: "))
    forecast_months = previous_months
    end_month_input = input("Enter the month you want to forecast (YYYY/MM format): ")

    # Split the input into year and month components
    end_year, end_month_val = map(int, end_month_input.split('/'))

    outlier_treated_columns = result_df.copy()
    for col in consumption_cols:
        outlier_treated_columns[f'ot_{col}'] = outlier_treated_columns[col]

    # Check if the end month is in the available date columns
    if end_month_input in consumption_cols:
        end_index = consumption_cols.index(end_month_input)
    else:
        print("Continuing with previous months.")
        end_index = len(consumption_cols)-1

    if end_month_input in consumption_cols:
        start_index = max(0, end_index - previous_months)
    else:
        start_index = max(0, end_index - previous_months + 1)
    difference_columns = []

    for i in range(start_index, end_index+1):
        current_month = consumption_cols[i]
        current_month_index = consumption_cols.index(current_month)
        previous_months_indices = list(range(current_month_index - previous_months, current_month_index))
        previous_months_data = [consumption_cols[idx] for idx in previous_months_indices]

        selected_data = outlier_treated_columns[previous_months_data]
        consumptions = np.apply_along_axis(calculate_consumption, axis=1, arr=selected_data.values,
                                           window_size=previous_months)

        predicted_month = current_month
        column_name = f'predicted_consumption_{predicted_month.replace("/", "_")}'

        df[column_name] = consumptions
        if current_month != end_month_input:
        # Calculate difference column
            difference_column_name = f'Diff_{predicted_month}'
            difference = df[column_name] - outlier_treated_columns[f'ot_{predicted_month}']
            df[difference_column_name] = difference
            difference_columns.append(difference_column_name)

    # Define column_name_end_month before using it
    column_name_end_month = f'predicted_consumption_{end_month_input.replace("/", "_")}'

    if end_month_input not in consumption_cols:
        end_month_prediction_data = outlier_treated_columns[
            consumption_cols[-previous_months:]]  # Use outlier treated columns for prediction
        consumptions_end_month = np.apply_along_axis(calculate_consumption, axis=1,
                                                     arr=end_month_prediction_data.values, window_size=previous_months)
    else:
        end_input_index = consumption_cols.index(end_month_input)
        start_index = max(0, end_input_index - previous_months)

        # Select the columns for the last previous_months before end_input_index
        selected_data = outlier_treated_columns[consumption_cols[start_index:end_input_index]]

        # Calculate consumption for the selected data
        consumptions_end_month = np.apply_along_axis(calculate_consumption, axis=1, arr=selected_data.values,
                                                     window_size=previous_months)

    # Now you can safely use column_name_end_month
    df[column_name_end_month] = consumptions_end_month

    # Calculate modified forecasting for the end month
    difference_avg = df[difference_columns].mean(axis=1)
    modified_forecasting_end_month = df[column_name_end_month] - difference_avg
    modified_forecasting_end_month = np.where(modified_forecasting_end_month < 0, 0,
                                              modified_forecasting_end_month)  # Set negative values to zero
    modified_forecasting_end_month = np.round(modified_forecasting_end_month, 0)  # Round off to zero decimal places
    df[f'modified_forecasting_{end_month_input.replace("/", "_")}'] = modified_forecasting_end_month

    # Calculate safety stock
    lead_time = df['leadtime']
    safety_stock = np.where(lead_time <= 30, modified_forecasting_end_month,
                            modified_forecasting_end_month * lead_time / 30)
    df['safety_stock'] = safety_stock

    # Calculate cost of safety stock
    cost_of_safety_stock = df['safety_stock'] * df['Moving price']
    df['cost_of_safety_stock'] = cost_of_safety_stock
    last_month_safety_stock = df['last month safety stock']  # Assume the last column in consumption_cols is the last month's safety stock
    df['diff_safety_stock'] = df['safety_stock'] - last_month_safety_stock
    # Round off all columns to zero decimal places
    df = df.round(0)
    warnings.filterwarnings("ignore")
    output_df = df[
        ['Material', 'Material Description', 'leadtime', 'Moving price','last month safety stock'] + consumption_cols +
        [f'predicted_consumption_{end_month_input.replace("/", "_")}',
         f'modified_forecasting_{end_month_input.replace("/", "_")}',
         'safety_stock', 'cost_of_safety_stock','diff_safety_stock']
        ]
    # Save the output DataFrame to an Excel file
    output_file = "D:/final_output_modified_forecasting.xlsx"
    output_df.to_excel(output_file, index=False)
    print(f"Forecasting done for {end_month_input}")

if __name__ == "__main__":
    main()