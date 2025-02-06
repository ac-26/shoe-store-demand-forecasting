import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)

#Put your path
data = pd.read_excel('/Users/arnavchopra/Desktop/PE-Data Mining Shoe Dataset.xlsx', index_col=False)
print("\n\n\nSample dataset :- \n\n", data.head() )

print("\n\n\nShape of the dataset = ", end="")
print( data.shape)

print("\n\n\n Sample data decription : \n")
print( data.describe() )

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

print(data.isnull().sum())















