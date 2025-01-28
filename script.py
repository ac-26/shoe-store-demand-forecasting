import pandas as pd
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.max_rows', None)     # Show all rows (or set a number like 100 if you want to limit)
pd.set_option('display.width', None)        #Line sppacing

#Starting with data_cleaning and EDA(Exploratory Data Analysis)

#loading the dataset in script in "data" variable
# @ashu When loading in your laptop change file path
data = pd.read_excel('/Users/arnavchopra/Desktop/PE-Data Mining Shoe Dataset.xlsx', index_col = False)

#printing sample
print("Sample Shoe Sales Dataset :- \n")
print(data.head())
print("\n")

print("Shape of the Dataset = ")
print(data.shape)
print("\n")

print("Dataset Description :- \n")
print(data.describe())
print("\n")

# Columns like Barcode can be deleted and are not required for Data Mining and training a model as Item No. is already given.
data = data.drop('BARCODE', axis = 1)


#Checking for missing/NULL values
print(data.isnull().sum())

#Working on Colour col for NULL values



##print( data[data['Colour'].isnull()]['Colour'] )


