import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)

##Put your path
data = pd.read_excel('/Users/arnavchopra/Desktop/PE-Data Mining Shoe Dataset.xlsx', index_col = False)
print("\n\n\nSample dataset :- \n\n", data.head() )

print("\n\n\nShape of the dataset = ", end="")
print( data.shape)

print("\n\n\n Sample data decription : \n")
print( data.describe() )

#According to project requirement we don't need Barcode and EAN column as it is of no help to us, item no. acts as primary key
data.drop('BARCODE', axis = 1, inplace = True)
data.drop('EAN', axis = 1, inplace = True)

#Finding out NULL values
print("\n\n",data.isnull().sum())

#----------------------
#Observing Color column
data['Colour'].value_counts().plot(kind='bar')
plt.title('Colour Distribution')
plt.xlabel('Colour')
plt.ylabel('Frequency')
#plt.show()

#For colour column we can use mode value to fill null positions  as it is skewed way too much
data['Colour'] = data['Colour'].fillna(data['Colour'].mode()[0])

print("\n\n",data.isnull().sum())
print("\n\n\n")

#----------------------
#Observing Size columns
print(data['SIZE'].unique())

data["SIZE"] = data["SIZE"].astype(str).str.strip().str.upper()  # Remove spaces & normalize case

print("\n\n", data['SIZE'].unique())

#Creating a mapping
# Shoe size to cm mapping based on US, UK, and EU sizes
# Example conversion dictionary (Modify based on actual size conversions)

# Cleaning incorrect formats like '(08'
data['SIZE'] = [str(size).strip("()") if isinstance(size, str) else size for size in data['SIZE']]

size_to_cm = {
    "38": 24.0, "36": 23.0, "5": 22.5, "8": 26.0, "08C": 15.0, "41": 26.5,
    "40": 25.5, "3": 21.5, "6": 23.5, "42": 27.0, "10": 28.0, "9": 27.5,
    "32": 20.0, "13C": 19.0, "12C": 18.5, "29": 17.0, "33": 20.5,
    "7C": 14.0, "7": 24.5, "2": 21.0, "39": 25.0, "43": 27.5,
    "11C": 17.5, "37": 23.5, "4": 22.0, "1": 20.5, "31": 19.5,
    "45": 29.0, "07C": 13.5, "30": 18.0, "44": 28.0, "11": 29.0,
    "10C": 16.5, "8C": 14.5, "9C": 15.5, "25": 15.0, "24": 14.5,
    "05C": 12.0, "09C": 15.0, "06C": 13.0, "35": 22.5, "26": 16.0,
    "27": 16.5, "28": 17.0, "23": 14.0, "46": 29.5, "04C": 11.5,
    "34": 21.0, "22": 13.5, "6C": 13.5, "5C": 12.5, "12": 30.0,
    "3C": 11.0, "4C": 11.5, "08" : 26.0, "07" : 24.5
}

#Map sizes to centimeters and fill missing values with "Unknown"
data["SIZE"] = data["SIZE"].map(size_to_cm).fillna("Unknown")


print("\n\n", data['SIZE'].unique())

#Mapping has been created and implemented to the data frame. Now we will handle null values.
#I have changed all NAN values to Unknown
#We will fill NAN values with mode(Categorical)

mode_value = data["SIZE"].mode()[0]

# Replace 'Unknown' with the mode
data["SIZE"] = data["SIZE"].replace("Unknown", mode_value)

print("\n\n", data['SIZE'].unique())

print("\n\n",data.isnull().sum())


#----------------------
#Observing P_GROUP column
print(data['P_GROUP'].unique())
#Finding correlation between footwear and accessories

#Footwear -> 1 and Accessories -> 0
data["P_GROUP_NUM"] = data["P_GROUP"].map({"Footwear": 1, "Accessories": 0})

correlation_value = data["P_GROUP_NUM"].corr(data["QTY"])

print(f"Correlation between P_GROUP and QTY: {correlation_value}")


















