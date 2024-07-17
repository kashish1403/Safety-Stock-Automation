import pandas as pd
import numpy as np
import re
import sys
import warnings

# Function to rearrange columns
def rearrange_columns(input_file):
    df = pd.read_excel(input_file)
    date_columns = [col for col in df.columns if isinstance(col, str) and '/' in col and len(col) == 7]
    if not date_columns:
        print("No date columns found in 'YYYY/MM' format in the DataFrame.")
        return df
    valid_date_columns = [col for col in date_columns if col.split('/')[0].isdigit() and col.split('/')[1].isdigit()]
    if not valid_date_columns:
        print("No valid date columns found.")
        return df
    latest_date_column = max(valid_date_columns, key=lambda x: (int(x.split('/')[0]), int(x.split('/')[1])))
    sorted_columns = sorted(valid_date_columns, key=lambda x: (int(x.split('/')[0]), int(x.split('/')[1])))
    sorted_columns.remove(latest_date_column)
    sorted_columns.append(latest_date_column)
    df_combined = df[sorted_columns]
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
def calculate_sma(numbers, window_size):
    return np.mean(numbers[-window_size:])

def main():
    if len(sys.argv) != 2:
        print("Usage: python file.py <input_file_path>")
        sys.exit(1)

    input_file = sys.argv[1]
    # input_file = "C:/Users/Admin/Downloads/input_file_new.xlsx"

    if not (os.path.isfile(input_file)):
        print("Error: One or both of the specified files do not exist.")
        sys.exit(1)
    df = rearrange_columns(input_file)
    df = replace_negatives_and_nan_with_zero(df)
    # Find quartiles and replace values

    consumption_cols = [col for col in df.columns if re.match(r'\d{4}/\d{2}', col)]
    consumption_cols = [
        col if not df[col].apply(lambda x: np.isnan(x) or x < 0).any() else df[col].apply(lambda x: max(0, x)) for col
        in consumption_cols]
    forecast_month = input("Enter the month you want to forecast (YYYY/MM format): ")
    previous_months = int(input("Enter the number of previous months to consider for SMA calculation (3, 6, or 12): "))

    # Ensure at least one of the previous months is present in the data
    valid_window_sizes = [3, 6, 12]
    if previous_months not in valid_window_sizes:
        print("Invalid window size. Please enter 3, 6, or 12.")
        return

    # Check if any of the previous months are available in the data
    valid_window_present = any(previous_months <= len(consumption_cols) for previous_months in valid_window_sizes)

    if not valid_window_present:
        print(f"None of the window sizes {valid_window_sizes} are present in the data.")
        return

    if forecast_month not in consumption_cols:
        # If forecast_month is not in the data, use the same logic as for other months
        forecast_data = df[consumption_cols[-previous_months:]]
        consumptions = np.apply_along_axis(calculate_sma, axis=1, arr=forecast_data.values, window_size=previous_months)
    else:
        # If forecast_month is in the data, calculate consumption accordingly
        forecast_month_index = consumption_cols.index(forecast_month)
        start_index = max(0, forecast_month_index - previous_months)
        selected_data = df[consumption_cols[start_index:forecast_month_index]]
        consumptions = np.apply_along_axis(calculate_sma, axis=1, arr=selected_data.values, window_size=previous_months)

    predicted_consumption_column = f'predicted_consumption_{forecast_month.replace("/", "_")}'
    df[predicted_consumption_column] = np.round(consumptions, 0)  # Round off to zero decimal places

    # Calculate safety stock and its cost
    lead_time = df['leadtime']
    safety_stock = np.where(lead_time <= 30, df[predicted_consumption_column], df[predicted_consumption_column] * lead_time / 30)
    df['safety_stock'] = np.round(safety_stock, 0)  # Round off to zero decimal places
    cost_of_safety_stock = df['safety_stock'] * df['Moving price']
    df['cost_of_safety_stock'] = np.round(cost_of_safety_stock, 0)  # Round off to zero decimal places
    last_month_safety_stock = df[
        'last month safety stock']  # Assume the last column in consumption_cols is the last month's safety stock
    df['diff_safety_stock'] = df['safety_stock'] - last_month_safety_stock
    df = df.round(0)
    output_file = "final_output_forecasting_sma.xlsx"
    df.to_excel(output_file, index=False)
    print(f"Forecasting done for {forecast_month}")

if __name__ == "__main__":
    main()
