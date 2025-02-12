from pyexpat import features

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)

#Put your path
data = pd.read_excel('/Users/arnavchopra/Desktop/Datasets/PE-Data Mining Shoe Dataset.xlsx', index_col=False)
# print("\n\n\nSample dataset :- \n\n", data.head() )

# print("\n\n\nShape of the dataset = ", end="")
print( data.shape)

# print("\n\n\n Sample data decription : \n")
# print( data.describe() )

# #According to project requirement we don't need Barcode and EAN column as it is of no help to us, item no. acts as primary key
data.drop('BARCODE', axis = 1, inplace = True)
data.drop('EAN', axis = 1, inplace = True)

#Finding out NULL values
# print("\n\n",data.isnull().sum())
#
# #----------------------
# #Observing Color column
data['Colour'].value_counts().plot(kind='bar')
plt.title('Colour Distribution')
plt.xlabel('Colour')
plt.ylabel('Frequency')
#plt.show()
#
# #For colour column we can use mode value to fill null positions  as it is skewed way too much
data['Colour'] = data['Colour'].fillna(data['Colour'].mode()[0])
#
# print("\n\n",data.isnull().sum())
# print("\n\n\n")
#
# #----------------------
# #Observing Size columns
# print(data['SIZE'].unique())
#
data["SIZE"] = data["SIZE"].astype(str).str.strip().str.upper()  # Remove spaces & normalize case
#
# print("\n\n", data['SIZE'].unique())
#
# #Creating a mapping
# # Shoe size to cm mapping based on US, UK, and EU sizes
# # Example conversion dictionary (Modify based on actual size conversions)
#
# # Cleaning incorrect formats like '(08'
data['SIZE'] = [str(size).strip("()") if isinstance(size, str) else size for size in data['SIZE']]
#
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

# #Map sizes to centimeters and fill missing values with "Unknown"
data["SIZE"] = data["SIZE"].map(size_to_cm).fillna("Unknown")
#
#
# print("\n\n", data['SIZE'].unique())
#
# #Mapping has been created and implemented to the data frame. Now we will handle null values.
# #I have changed all NAN values to Unknown
# #We will fill NAN values with mode(Categorical)
#
mode_value = data["SIZE"].mode()[0]
#
# # Replace 'Unknown' with the mode
data["SIZE"] = data["SIZE"].replace("Unknown", mode_value)
#
# print("\n\n", data['SIZE'].unique())
#
# print("\n\n",data.isnull().sum())


#----------------------
# #Observing P_GROUP column
# print(data['P_GROUP'].unique())
# #Finding correlation between footwear and accessories

 #Footwear -> 1 and Accessories -> 0
data["P_GROUP_NUM"] = data["P_GROUP"].map({"Footwear": 1, "Accessories": 0})

correlation_value = data["P_GROUP_NUM"].corr(data["QTY"])

# print(f"Correlation between P_GROUP and QTY: {correlation_value}")

#---------------------
#P_GROUP EDA

# print(data[data['P_GROUP'] == 'ACCESSORIES']['BRAND'].unique())

#If the Brand is LFO or LSL we know its a ACCESSORY

data.loc[data['P_GROUP'].isnull() & data['BRAND'].isin(['LFO', 'LSL']), 'P_GROUP'] = 'ACCESSORIES'

#If the BRAND is different from LFO or LSL we know its a footwear

data.loc[data['P_GROUP'].isnull() & ~data['BRAND'].isin(['LFO', 'LSL']), 'P_GROUP'] = 'FOOTWEAR'

# print(data.isnull().sum())

#----------------
#BRAND EDA
# print(data['BRAND'].unique())

#Using the same logic if the BRAND is None, and if we know that P_GROUP is ACCESSORY then we know that it is either LFO or LSL(we take the mode)

# print(data[data['BRAND'].isin(['LFO','LSL'])]['BRAND'].mode())

data.loc[data['BRAND'].isnull() & data['P_GROUP'].isin(['ACCESSORIES']), 'BRAND'] = 'LFO'


#And if the P_GROUP is footwear we know that is the companies other than LFO OR LSL(we take the mode between those all companies)

# print(data[~data['BRAND'].isin(['LFO','LSL'])]['BRAND'].mode())

data.loc[data['BRAND'].isnull() & data['P_GROUP'].isin(['FOOTWEAR']), 'BRAND'] = 'GLIDERS'

# print(data.isnull().sum())

#-------------------
#Observing Leather Type
# print(data[data['LETHR_TYPE'].isnull()].head())
#Filling the nan values of lethr type wherever p_group is accessories
#as not applicable as any accessory inherently wont have a leather type.
data.loc[data["P_GROUP"] == "ACCESSORIES", "LETHR_TYPE"] = "Not Applicable"

# print("\n\n\n",data[data['LETHR_TYPE'].isnull()].head())

#for the rest of the values as this is categorical data we use mode
data['LETHR_TYPE'] = data['LETHR_TYPE'].fillna(data['LETHR_TYPE'].mode()[0])

# print(data.isnull().sum())


#Handling spelling errors and uniformity
data.drop(columns=["P_GROUP_NUM"], inplace=True)
# print(data.info())



data.rename(columns={"Colour": "COLOUR", "P_GROUP": "ITEM_TYPE", "LETHR_TYPE": "LEATHER_TYPE"}, inplace=True)
# print(data.info())

#Making date format correct
# print(data['RECEIPT_DATE'].head())
data['RECEIPT_DATE'] = data['RECEIPT_DATE'].astype(str)
data['YEAR']=data['RECEIPT_DATE'].str.split('-').str[0]
data['MONTH']=data['RECEIPT_DATE'].str.split('-').str[1]
data['DATE']=data['RECEIPT_DATE'].str.split('-').str[2]

# print("\n\n\n", data.head())

#Reordering columns
last_col = data.columns[-1]
col_data = data.pop(last_col)
data.insert(0, last_col, col_data)

last_col = data.columns[-1]
col_data = data.pop(last_col)
data.insert(0, last_col, col_data)

last_col = data.columns[-1]
col_data = data.pop(last_col)
data.insert(0, last_col, col_data)


#Deleting the old date format col
data.drop(columns=["RECEIPT_DATE"], inplace=True)


#-------------------------
#Observing SEASON column

# print("\n\n\n", data.loc[data['SEASON'].isnull()].head())

# print("\n\n\n", data['SEASON'].unique())
# print("\n\n\n", data['YEAR'].unique())
#
# print("\n\n",data.loc[data['SEASON'] == 'S11'].head())

#Checking which have most missing values in season, accessories or footwear
#print("\n\n\n" ,data.groupby("ITEM_TYPE")["SEASON"].apply(lambda x: x.isnull().sum()))
#Both are equal in null values

# #Checking article name that have null season
# print(data[data["SEASON"].isnull()]["ARTICLE_NAME"].value_counts())

# print(data.loc[data['ARTICLE_NAME'] == "TRENDY"])
# Filter data where ARTICLE_NAME is "TRENDY"
#Categorical analysis of seasons for article name == trendy
# trendy_data = data[data["ARTICLE_NAME"] == "TRENDY"]
# # Counting occurrences of each SEASON category
# season_counts = trendy_data["SEASON"].value_counts()
# print(season_counts)

#We observe that maximum times season is SUMMER so we fill na values in season for article name trendy to summer.
data.loc[(data["ARTICLE_NAME"] == "TRENDY") & (data["SEASON"].isnull()), "SEASON"] = "SUMMER"

#print(data.loc[data['ARTICLE_NAME'] == "WP-60"])
#Filter data where ARTICLE_NAME is "WP-60"


#As this is accessory season should be universal
data.loc[ (data["ITEM_TYPE"] == "ACCESSORIES") & (data["SEASON"].isnull()), "SEASON"] = "UNIVRSL"
#VERIFY
#print(data[data["ITEM_TYPE"] == "ACCESSORIES"]["SEASON"].isnull().sum())  # Should print 0
#All accessories season handled


# trendy_data = data[data["ARTICLE_NAME"] == "S/BOY EXCE"]
# # Counting occurrences of each SEASON category
# season_counts = trendy_data["SEASON"].value_counts()
# print(season_counts)

# print(data.loc[ (data['ARTICLE_NAME'] == 'S/BOY EXCE') & (data['SEASON'] == "UNIVRSL") ])

#Filling S/BOY EXCE season with UNIVRSL because max of it is UNIVRSL
data.loc[ (data["ARTICLE_NAME"] == "S/BOY EXCE") & (data["SEASON"].isnull()), "SEASON"] = "UNIVRSL"
#VERIFY
#print(data[data["ARTICLE_NAME"] == "S/BOY EXCE"]["SEASON"].isnull().sum())  # Should print 0



# trendy_data = data[data["ARTICLE_NAME"] == "S/BOY-V"]
# # Counting occurrences of each SEASON category
# season_counts = trendy_data["SEASON"].value_counts()
# print(season_counts)
#
# print( "\n\n" ,data.loc[ (data['ARTICLE_NAME'] == 'S/BOY-V') & (data['SEASON'] == "UNIVRSL") ].head(10))
#
# print( "\n\n" ,data.loc[ (data['ARTICLE_NAME'] == 'S/BOY-V') & (data['SEASON'] == "SS") ].head(10))

#We observe there is a 50-50 split in universal and ss so we fill ss and universal randomly to maintain consistency
possible_values = ["UNIVRSL", "SS"]
data.loc[(data["ARTICLE_NAME"] == "S/BOY-V") & (data["SEASON"].isnull()), "SEASON"] = np.random.choice(possible_values)

#VERIFY
# print(data[data["ARTICLE_NAME"] == "S/BOY-V"]["SEASON"].isnull().sum())  # Should print 0

# trendy_data = data[data["ARTICLE_NAME"] == "LB-16"]
# # Counting occurrences of each SEASON category
# season_counts = trendy_data["SEASON"].value_counts()
# print(season_counts)

#ALL values of season for lb-16 is null so we use universal
# print(data.loc[( data["ARTICLE_NAME"] == "LB-16") ].head(20))

data.loc[ (data["ARTICLE_NAME"] == "LB-16") & (data["SEASON"].isnull()), "SEASON"] = "UNIVRSL"
#VERIFY
# print(data[data["ARTICLE_NAME"] == "LB-16"]["SEASON"].isnull().sum())  # Should print 0

# print(data[data["SEASON"].isnull()]["ARTICLE_NAME"].value_counts())

# trendy_data = data[data["ARTICLE_NAME"] == "3070-27"]
# # Counting occurrences of each SEASON category
# season_counts = trendy_data["SEASON"].value_counts()
# print(season_counts)

#We obvserve AW is more so we put AW in season
data.loc[ (data["ARTICLE_NAME"] == "3070-27") & (data["SEASON"].isnull()), "SEASON"] = "AW"
#VERIFY
# print(data[data["ARTICLE_NAME"] == "3070-27"]["SEASON"].isnull().sum())  # Should print 0

# print(data[data["SEASON"].isnull()]["ARTICLE_NAME"].value_counts())

# trendy_data = data[data["ARTICLE_NAME"] == "DR-519"]
# # Counting occurrences of each SEASON category
# season_counts = trendy_data["SEASON"].value_counts()
# print(season_counts)

data.loc[ (data["ARTICLE_NAME"] == "DR-519") & (data["SEASON"].isnull()), "SEASON"] = "SUMMER"
#VERIFY
# print(data[data["ARTICLE_NAME"] == "DR-519"]["SEASON"].isnull().sum())  # Should print 0


# print(data[data["SEASON"].isnull()]["ARTICLE_NAME"].value_counts())


#This returns the mode value
def safe_mode(x):
    mode = x.mode()
    if not mode.empty:
        return mode.iloc[0]  # Return the first mode if multiple modes exist
    return 'UNIVRSL'  #Return 'UNIVERSAL' if no mode exists (i.e., all values are unique or NaN)

# Replace NaN values in SEASON with the mode for each ARTICLE_NAME
def fill_season_mode(data):

    mode_seasons = data.groupby('ARTICLE_NAME')['SEASON'].agg(safe_mode).reset_index()


    data = data.merge(mode_seasons, on='ARTICLE_NAME', how='left', suffixes=('', '_mode'))

    # Fill NaN values in SEASON with the corresponding mode from the merged column
    data['SEASON'] = data['SEASON'].fillna(data['SEASON_mode'])


    data.drop(columns=['SEASON_mode'], inplace=True)

    return data

data = fill_season_mode(data)

# Verify
# print(data[data['SEASON'].isnull()]["ARTICLE_NAME"].value_counts())

# print(data['SEASON'].unique())

#---------------
#Observing Indicator
# print(data['INDICATOR'].unique())

data.drop(columns=["INDICATOR"], inplace=True)

# print(data.isnull().sum())

# print(data['SEASON'].unique())

data['SEASON'] = data['SEASON'].replace({
    "S10": "SUMMER",
    "S11": "SUMMER",
    "S12": "SUMMER",
    "S13": "SUMMER",
    "S14": "SUMMER",
    "S15": "SUMMER",
    "S16": "SUMMER",
    "S17": "SUMMER",
    "S18": "SUMMER",
    "S19": "SUMMER",
    "S20": "SUMMER",
    "SS": "SUMMER",
    "W10": "WINTER",
    "W11": "WINTER",
    "W12": "WINTER",
    "W13": "WINTER",
    "W14": "WINTER",
    "W15": "WINTER",
    "W16": "WINTER",
    "W17": "WINTER",
    "W18": "WINTER",
    "W19": "WINTER",
    "W20": "WINTER",
    "AW": "WINTER",
})

#For now I have taken AW as winter, and I have left universal


# print(data['SEASON'].unique())

# print((data['SEASON'] == 'UNIVRSL').sum())

#Cleaning the Colour Section

data["COLOUR"] = data["COLOUR"].replace({
    "OLGREEN" : "OLIVE_GREEN",
    "D.GREY" : "DARK_GREY",
    "R.BLUE": "RED BLUE",
    "P.GREEN": "PALE GREEN",
    "T.BLUE": "TURQUOISE BLUE",
    "N.BLUE": "NAVY BLUE",
    "L.BEIGE": "LIGHT BEIGE",
    "Maroon": "MAROON",
    "MAHROON": "MAROON",
    "purple": "PURPLE",
    "violet": "VIOLET",
    "S.BLUE": "SKY BLUE",
    "elephnt": "ELEPHANT",
    "ELEPHNT": "ELEPHANT",
    "S.GREEN": "SAGE GREEN",
    "natural": "NATURAL",
    "golden": "GOLDEN",
    "cream": "CREAM",
    "D.BROWN": "DARK BROWN",
    "L.GREY": "LIGHT GREY",
    "Black": "BLACK",
    "black": "BLACK",
    "ASSTD": "ASSORTED",
    "ASST": "ASSORTED",
    "asstd": "ASSORTED",
    "assorted": "ASSORTED",
    "OFFWHITE": "OFF WHITE",
    "GNMETAL": "GUN METAL",
    "REVERSAB": "REVERSIBLE",
    "REVERSEB" :  "REVERSIBLE",
    "Reversib" :  "REVERSIBLE",
    "Brown": "BROWN",
    "brown" : "BROWN",
    "lemon": "LEMON",
    "L.GREEN": "LIGHT GREEN",
    "fawn": "FAWN",
    "neutral": "NEUTRAL",
    "skin": "SKIN",
    "silver": "SILVER",
    "beige": "BEIGE",
    "Beige": "BEIGE",
    "multipl": "MULTI",
    "MULTIPL": "MULTI",
    "khaki": "KHAKI",
    "Blue": "BLUE",
    "Navy": "NAVY",
    "skin+bk": "SKIN BLACK",
    "Assorted": "ASSORTED",
    "Tan": "TAN",
    "L.BROWN": "LIGHT BROWN",
    "Camel": "CAMEL",
    "toupe": "TOUPE",
    "Cherry": "CHERRY",
    "Dark Bro": "DARK BROWN",
    "Grey": "GREY",
    "mauve": "MAUVE",
    "mustard": "MUSTARD",
    "peach": "PEACH",
    "PEAH": "PEACH",
    "Red": "RED",
})

# print(data["COLOUR"].unique())

# print(data["SEASON"].unique())


# print(data.head())

# Cleaning GROSS_VALUE column

# print(data["GROSS_VALUE"].unique())

data["GROSS_VALUE"] = data["GROSS_VALUE"].apply(lambda x: abs(x) if str(x).startswith('-') else x)

# print(data["GROSS_VALUE"].unique())



#Checking where gross value is zero
# print(data['ITEM_TYPE'].loc[data['GROSS_VALUE'] == 0])
indexZeroPrice = data[ (data['GROSS_VALUE'] == 0) ].index
data.drop(indexZeroPrice , inplace=True)
# data.head(15)
#Verifying
# print(data['ITEM_TYPE'].loc[data['GROSS_VALUE'] == 0])

#There are many items where quantity is negative
# print(data.loc[data['QTY'] <= 0])
indexZeroQTY = data[ (data['QTY'] <= 0) ].index
data.drop(indexZeroQTY , inplace=True)
# data.head(15)
#Verifying
# print(data.loc[data['QTY'] <= 0])

#There are rows where discount is negative
# print(data.loc[data['DIS%'] < 0])
indexZeroDisc = data[ (data['DIS%'] < 0)].index
data.drop(indexZeroDisc , inplace=True)

#There are rows where discount is 100
# print(data.loc[data['DIS%'] == 100])
indexFullDisc = data[ (data['DIS%'] == 100)].index
data.drop(indexFullDisc , inplace=True)
#Verifying
# print(data.loc[data['DIS%'] == 100])

#There are values where net_value is 0
# print(data.loc[data['NET_VALUE'] == 0])
indexZeroPrice = data[ (data['NET_VALUE'] == 0)].index
data.drop(indexZeroPrice , inplace=True)
#Verifying
# print(data.loc[data['DIS%'] == 100])


# print(data[data["GROSS_VALUE"] == 0].shape[0])
# print(data[data["GROSS_VALUE"] == 0])

# print(data.loc[data['QTY'] == min(data['QTY'])])
# print("\n\n")
# # print(data['BRAND'].unique())
# print(data.shape)

# print(data.columns)
# print(data['ITEM'].unique().size)



#Multiple item number exist multiple times so this is also categorical
# print(data.loc[data['ITEM'] == 57390])

#To apply ANN we need all categorical columns into numbers so we use encoding techniques
from sklearn.preprocessing import LabelEncoder

# For High-Cardinality Categorical Features → Label Encoding is used
categorical_cols = ['ARTICLE_NAME', 'ITEM', 'BRAND', 'COLOUR', 'LEATHER_TYPE']

for col in categorical_cols:
    data[col] = data[col].astype(str)  # Convert to string

# Now apply Label Encoding
from sklearn.preprocessing import LabelEncoder

label_encoders = {}

for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    data[col] = label_encoders[col].fit_transform(data[col])


ITEM_TYPE_MAPPING = {
    "FOOTWEAR" : 0,
    "ACCESSORIES" : 1,
}

data["ITEM_TYPE"] = data["ITEM_TYPE"].map(ITEM_TYPE_MAPPING)

data["SEASON"] = data["SEASON"].replace({"UNIVRSL" : "UNIVERSAL"})

SEASON_MAP ={
    "SUMMER": 0,
    "WINTER" : 1,
    "UNIVERSAL" : 2,
}

data["SEASON"] = data["SEASON"].map(SEASON_MAP)

# print(data.head(25))

# print("\n\n\n Sample data decription : \n")
# print( data.describe() )

#-----------------
#Scaling numerical features

#Before scaling we check if it is normally distributed or not
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


numerical_cols = ["YEAR", "MONTH", "DATE", "QTY", "GROSS_VALUE", "DIS%", "DISC_VALUE", "NET_VALUE"]

# # Plot histograms
# for col in numerical_cols:
#     plt.figure(figsize=(6, 4))
#     sns.histplot(data[col], kde=True)
#     plt.title(f"Histogram of {col}")
#     plt.show()
#
# # Q-Q Plot for checking normality
# for col in numerical_cols:
#     plt.figure(figsize=(6, 4))
#     stats.probplot(data[col], dist="norm", plot=plt)
#     plt.title(f"Q-Q Plot of {col}")
#     plt.show()

#Because it is not clear from visualization we will do statisticly
# from scipy.stats import shapiro
#
# for col in numerical_cols:
#     stat, p = shapiro(data[col].dropna())
#     print(f"{col}: p-value = {p:.5f}")
#
#     if p > 0.05:
#         print("Likely Normally Distributed\n")
#     else:
#         print("Not Normally Distributed\n")

#We find out that it is not normally distributed hence we can use MinMax Scaling
from sklearn.preprocessing import MinMaxScaler

# Initializing MinMaxScaler
scaler = MinMaxScaler()
# Fitting and transforming numerical columns
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# print(data.head())

