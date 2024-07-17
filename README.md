# Safety-Stock-Automation
In "Safety Stock Automation," we automate material safety stock calculations using Python. Forecasting methods like time series and moving averages optimize procedures, aiming for complete automation without human involvement.
# Process
Input:
Historical data from Excel files containing consumption and lead time information.
User input to select the historical data range (3, 6, 9, 12, or 24 months).
Method/Approach:
Preprocess the input data using Python's pandas library.
Apply time series forecasting methods (SMA, WMA, Exponential Smoothing) to predict future consumption.
Implement algorithms to dynamically calculate safety stock based on the processed data and user-selected historical range.
Automate the entire process to ensure consistent and accurate safety stock calculations.
Output:
An automated Excel file generated using openpyxl.
The output file includes predicted consumption and recommended safety stock levels for each material.
The automation reduces manual effort in safety stock calculation by lead time by 50%, enhancing operational efficiency and accuracy.
Tools and Technologies Used: Python, pandas, openpyxl, Time Series Analysis, SMA, WMA, Exponential Smoothing.
