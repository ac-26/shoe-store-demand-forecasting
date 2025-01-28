import pandas as pd
import numpy as ns
import matplotlib.pyplot as plt

pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('display.width',None)

##Put your path
data = pd.read_excel('/Users/arnavchopra/Desktop/PE-Data Mining Shoe Dataset.xlsx', index_col = False)
print("\n\n\nSample dataset :- \n\n", data.head(5) )

print("\n\n\nShape of the dataset = ", end="")
print( data.shape)

print("\n\n\n Sample data decription : \n")
print( data.describe() )

#According to project requirement we don't need Barcode and EAN column as it is of no help to us, item no. acts as primary key
data.drop('BARCODE', axis = 1, inplace = True)
data.drop('EAN', axis = 1, inplace = True)

#Finding out NULL values
print("\n\n",data.isnull().sum())

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

'''
Leaving from SIZE @ashu
'''









