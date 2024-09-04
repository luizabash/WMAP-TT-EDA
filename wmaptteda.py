import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr

# Load the WMAP TT Power Spectrum data
file_path = 'wmap_tt_spectrum_5yr_v3p1.txt'

# Read the data into a pandas DataFrame, skipping the header lines that start with '#'
wmap_data = pd.read_csv(file_path, delim_whitespace=True, comment='#', header=None)

# Assign column names
wmap_data.columns = ['Multipole_Moment_l', 'TT_Power_Spectrum_uK2', 'Error_Fisher_uK2', 
                     'Measurement_Errors_uK2', 'Cosmic_Variance_uK2']

# Step 2: Summary Statistics
# Calculate summary statistics for each column
summary_stats = wmap_data.describe()
print("Summary Statistics:\n", summary_stats)

# Step 3: Data Visualization
# Plotting TT Power Spectrum against Multipole Moment (l)
plt.figure(figsize=(10, 6))
plt.plot(wmap_data['Multipole_Moment_l'], wmap_data['TT_Power_Spectrum_uK2'], label='TT Power Spectrum (uK^2)', color='blue')
plt.xlabel('Multipole Moment (l)')
plt.ylabel('TT Power Spectrum (uK^2)')
plt.title('WMAP TT Power Spectrum vs. Multipole Moment (l)')
plt.grid(True)
plt.legend()
plt.show()

# Plotting Errors against Multipole Moment (l)
plt.figure(figsize=(10, 6))
plt.plot(wmap_data['Multipole_Moment_l'], wmap_data['Error_Fisher_uK2'], label='Error from Fisher Matrix (uK^2)', color='red')
plt.plot(wmap_data['Multipole_Moment_l'], wmap_data['Measurement_Errors_uK2'], label='Measurement Errors (uK^2)', color='green')
plt.plot(wmap_data['Multipole_Moment_l'], wmap_data['Cosmic_Variance_uK2'], label='Cosmic Variance (uK^2)', color='orange')
plt.xlabel('Multipole Moment (l)')
plt.ylabel('Error (uK^2)')
plt.title('Errors vs. Multipole Moment (l)')
plt.grid(True)
plt.legend()
plt.show()

# Step 4: Model Fitting (Polynomial Regression)
# Preparing the data for polynomial regression
X = wmap_data[['Multipole_Moment_l']]
y = wmap_data['TT_Power_Spectrum_uK2']

# Define a polynomial of degree 5 for fitting
poly = PolynomialFeatures(degree=5)
X_poly = poly.fit_transform(X)

# Fit the polynomial regression model
model = LinearRegression()
model.fit(X_poly, y)

# Predict using the fitted model
y_pred = model.predict(X_poly)

# Calculate Mean Squared Error (MSE) to evaluate the model
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error (MSE):", mse)

# Plot the original data and the fitted polynomial curve
plt.figure(figsize=(10, 6))
plt.scatter(wmap_data['Multipole_Moment_l'], wmap_data['TT_Power_Spectrum_uK2'], color='blue', label='Original Data', alpha=0.5)
plt.plot(wmap_data['Multipole_Moment_l'], y_pred, color='red', label='Fitted Polynomial (Degree 5)', linewidth=2)
plt.xlabel('Multipole Moment (l)')
plt.ylabel('TT Power Spectrum (uK^2)')
plt.title('Polynomial Regression Fit to WMAP TT Power Spectrum')
plt.grid(True)
plt.legend()
plt.show()

# Step 5: Hypothesis Testing
# Calculate Pearson's correlation coefficient and p-value
pearson_corr, p_value = pearsonr(wmap_data['Multipole_Moment_l'], wmap_data['TT_Power_Spectrum_uK2'])
print("Pearson Correlation Coefficient:", pearson_corr)
print("P-value:", p_value)
