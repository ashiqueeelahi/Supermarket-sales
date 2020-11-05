import pandas as pd
import numpy as np;
import matplotlib.pyplot as plt;
import seaborn as sns;
from sklearn.impute import SimpleImputer;
from sklearn.compose import ColumnTransformer;
from sklearn.pipeline import Pipeline;
from sklearn.preprocessing import LabelEncoder;
from sklearn.preprocessing import StandardScaler;
from sklearn.preprocessing import MinMaxScaler;
from sklearn.model_selection import train_test_split;
from sklearn.linear_model import LinearRegression ;
from sklearn.linear_model import Ridge, Lasso;
from sklearn.metrics import mean_squared_error;
from sklearn.metrics import r2_score;
from sklearn.preprocessing import PolynomialFeatures;
from sklearn.svm import SVR;
from sklearn.svm import SVC;
from sklearn.tree import DecisionTreeClassifier;
from sklearn.ensemble import RandomForestClassifier;
from sklearn.ensemble import RandomForestRegressor;
from sklearn.neighbors import KNeighborsClassifier;
from sklearn.naive_bayes import GaussianNB;
import pickle;

Let us import the data first

data = pd.read_csv('../input/supermarket-sales/supermarket_sales - Sheet1.csv')

How does it look? Let's figure out

data.head(5)

	Invoice ID 	Branch 	City 	Customer type 	Gender 	Product line 	Unit price 	Quantity 	Tax 5% 	Total 	Date 	Time 	Payment 	cogs 	gross margin percentage 	gross income 	Rating
0 	750-67-8428 	A 	Yangon 	Member 	Female 	Health and beauty 	74.69 	7 	26.1415 	548.9715 	1/5/2019 	13:08 	Ewallet 	522.83 	4.761905 	26.1415 	9.1
1 	226-31-3081 	C 	Naypyitaw 	Normal 	Female 	Electronic accessories 	15.28 	5 	3.8200 	80.2200 	3/8/2019 	10:29 	Cash 	76.40 	4.761905 	3.8200 	9.6
2 	631-41-3108 	A 	Yangon 	Normal 	Male 	Home and lifestyle 	46.33 	7 	16.2155 	340.5255 	3/3/2019 	13:23 	Credit card 	324.31 	4.761905 	16.2155 	7.4
3 	123-19-1176 	A 	Yangon 	Member 	Male 	Health and beauty 	58.22 	8 	23.2880 	489.0480 	1/27/2019 	20:33 	Ewallet 	465.76 	4.761905 	23.2880 	8.4
4 	373-73-7910 	A 	Yangon 	Normal 	Male 	Sports and travel 	86.31 	7 	30.2085 	634.3785 	2/8/2019 	10:37 	Ewallet 	604.17 	4.761905 	30.2085 	5.3

Looks Perfect. No missing value what so ever. We can start to visualize it

data['Branch'].unique()

array(['A', 'C', 'B'], dtype=object)

Branch with Pieplot?

plt.figure(figsize = (16,9))
Branch = data.Branch.value_counts().reset_index()
plt.pie(Branch.Branch, labels = Branch['index'],autopct='%1.1f%%')
plt.title("Sales From Different Branches")
plt.show()

City Next

data['City'].unique()

array(['Yangon', 'Naypyitaw', 'Mandalay'], dtype=object)

plt.figure(figsize = (16,9))
City = data.City.value_counts().reset_index()
plt.pie(City.City, labels = City['index'],autopct='%1.1f%%')
plt.title("Sales From Different Cities")
plt.show()

Hmm! How can we forget the customers?

data['Customer type'].unique()

array(['Member', 'Normal'], dtype=object)

plt.figure(figsize = (16,9))
Customer = data['Customer type'].value_counts().reset_index()
plt.pie(Customer['Customer type'], labels = Customer['index'],autopct='%1.1f%%')
plt.title("Sales From Different Type of Customers")
plt.show()

Genders?

data['Gender'].unique()	

array(['Female', 'Male'], dtype=object)

plt.figure(figsize = (16,9))
Gender = data.Gender.value_counts().reset_index()
plt.pie(Gender.Gender, labels = Gender['index'],autopct='%1.1f%%')
plt.title("Sales From Different Genders")
plt.show()

Women are clearly winning here.
So, what kind of products people buy the most?

data['Product line'].unique()

array(['Health and beauty', 'Electronic accessories',
       'Home and lifestyle', 'Sports and travel', 'Food and beverages',
       'Fashion accessories'], dtype=object)

plt.figure(figsize = (20,10))

Product = data.groupby('Product line').size().to_frame(name = "count").reset_index()
sns.barplot(y = 'count', x='Product line', data = Product )

plt.title("Sales of Different Kinds of Products")
plt.xlabel("Product line")
plt.ylabel("Count")

plt.show()

data.head(3)

	Invoice ID 	Branch 	City 	Customer type 	Gender 	Product line 	Unit price 	Quantity 	Tax 5% 	Total 	Date 	Time 	Payment 	cogs 	gross margin percentage 	gross income 	Rating
0 	750-67-8428 	A 	Yangon 	Member 	Female 	Health and beauty 	74.69 	7 	26.1415 	548.9715 	1/5/2019 	13:08 	Ewallet 	522.83 	4.761905 	26.1415 	9.1
1 	226-31-3081 	C 	Naypyitaw 	Normal 	Female 	Electronic accessories 	15.28 	5 	3.8200 	80.2200 	3/8/2019 	10:29 	Cash 	76.40 	4.761905 	3.8200 	9.6
2 	631-41-3108 	A 	Yangon 	Normal 	Male 	Home and lifestyle 	46.33 	7 	16.2155 	340.5255 	3/3/2019 	13:23 	Credit card 	324.31 	4.761905 	16.2155 	7.4
Let's see the unit prices fractuation as well as ranges

plt.figure(figsize = (16,9))
b  = [10,20,30,40,50,60,70,80,90,100,110,120,130,140,150];
sns.distplot(a= data['Unit price'], bins = b, color ='Y')
plt.title("Unit Price of Different Kinds of Products")
plt.xlabel("Unit Price")
plt.ylabel("Count")

plt.show()

Tax ranges look like:

plt.figure(figsize = (16,9))
bb = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
sns.distplot(a= data['Tax 5%'], bins =bb, color = 'R')
plt.title("Tax of Different Kinds of Products")
plt.xlabel("Tax")
plt.ylabel("Count")

plt.show()

How much total sales look like?

plt.figure(figsize = (16,9))
sns.kdeplot(data= data['Total'], shade = True, color = 'G')
plt.title("Total Sales")
plt.xlabel("Total Sales")
plt.ylabel("Count")

plt.show()

Payment methods:

data['Payment'].unique()

array(['Ewallet', 'Cash', 'Credit card'], dtype=object)

Payment = data.groupby('Payment').size().to_frame(name = "count").reset_index()
sns.barplot(y = 'count', x='Payment', data = Payment )

plt.title("Payment Method")
plt.xlabel("Payment")
plt.ylabel("Count")

plt.show()

plt.figure(figsize = (16,9))
Payment = data.Payment.value_counts().reset_index()
plt.pie(Payment.Payment, labels = Payment['index'],autopct='%1.1f%%')
plt.title("Payment Method")
plt.show()

Cogs

plt.figure(figsize = (16,9))
sns.distplot(a= data['cogs'],  color = 'R')
plt.title("Cogs")
plt.xlabel("cogs")
plt.ylabel("Count")

plt.show()

Quantity:

data['Quantity'].unique()

array([ 7,  5,  8,  6, 10,  2,  3,  4,  1,  9])

plt.figure(figsize = (20,10))

Quantity = data.groupby('Quantity').size().to_frame(name = "count").reset_index()
sns.barplot(y = 'count', x='Quantity', data = Quantity )

plt.title("Quantity of Products Sold")
plt.xlabel("Quantity")
plt.ylabel("Count")

plt.show()

RATINGS SPEAK FOR THE CUSTOMERS. SO, HOW DO THEY RESPOND?

plt.figure(figsize = (16,9))
sns.kdeplot(data= data['Rating'], shade = True, color = 'G')
plt.title("Rating")
plt.xlabel("Rating")
plt.ylabel("Count")

plt.show()

Gross Income

plt.figure(figsize = (16,9))
bbB = [5,10,15,20,25,30,35,40,45,50,55,60]
sns.distplot(a= data['gross income'], bins =bbB, color = 'G')
plt.title("Gross Income")
plt.xlabel("gross income")
plt.ylabel("Count")

plt.show()

data['gross margin percentage'] = data['gross margin percentage'].astype('float64')

data

	Invoice ID 	Branch 	City 	Customer type 	Gender 	Product line 	Unit price 	Quantity 	Tax 5% 	Total 	Date 	Time 	Payment 	cogs 	gross margin percentage 	gross income 	Rating
0 	750-67-8428 	A 	Yangon 	Member 	Female 	Health and beauty 	74.69 	7 	26.1415 	548.9715 	1/5/2019 	13:08 	Ewallet 	522.83 	4.761905 	26.1415 	9.1
1 	226-31-3081 	C 	Naypyitaw 	Normal 	Female 	Electronic accessories 	15.28 	5 	3.8200 	80.2200 	3/8/2019 	10:29 	Cash 	76.40 	4.761905 	3.8200 	9.6
2 	631-41-3108 	A 	Yangon 	Normal 	Male 	Home and lifestyle 	46.33 	7 	16.2155 	340.5255 	3/3/2019 	13:23 	Credit card 	324.31 	4.761905 	16.2155 	7.4
3 	123-19-1176 	A 	Yangon 	Member 	Male 	Health and beauty 	58.22 	8 	23.2880 	489.0480 	1/27/2019 	20:33 	Ewallet 	465.76 	4.761905 	23.2880 	8.4
4 	373-73-7910 	A 	Yangon 	Normal 	Male 	Sports and travel 	86.31 	7 	30.2085 	634.3785 	2/8/2019 	10:37 	Ewallet 	604.17 	4.761905 	30.2085 	5.3
... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	... 	...
995 	233-67-5758 	C 	Naypyitaw 	Normal 	Male 	Health and beauty 	40.35 	1 	2.0175 	42.3675 	1/29/2019 	13:46 	Ewallet 	40.35 	4.761905 	2.0175 	6.2
996 	303-96-2227 	B 	Mandalay 	Normal 	Female 	Home and lifestyle 	97.38 	10 	48.6900 	1022.4900 	3/2/2019 	17:16 	Ewallet 	973.80 	4.761905 	48.6900 	4.4
997 	727-02-1313 	A 	Yangon 	Member 	Male 	Food and beverages 	31.84 	1 	1.5920 	33.4320 	2/9/2019 	13:22 	Cash 	31.84 	4.761905 	1.5920 	7.7
998 	347-56-2442 	A 	Yangon 	Normal 	Male 	Home and lifestyle 	65.82 	1 	3.2910 	69.1110 	2/22/2019 	15:33 	Cash 	65.82 	4.761905 	3.2910 	4.1
999 	849-09-3807 	A 	Yangon 	Member 	Female 	Fashion accessories 	88.34 	7 	30.9190 	649.2990 	2/18/2019 	13:28 	Cash 	618.38 	4.761905 	30.9190 	6.6

1000 rows × 17 columns
Want Some Mashup?
Which Branches sell what kind of products on how amnount?

plt.figure(figsize = (16,9))
sns.countplot(y ='Product line', hue = "Branch", data = data) 
plt.xlabel('Count')
plt.ylabel('Product Type')
plt.show()

Which Products are bought more by males and females?

plt.figure(figsize = (16,9))
sns.countplot(y ='Product line', hue = "Gender", data = data) 
plt.xlabel('Count')
plt.ylabel('Product Type')
plt.show()

Which branch has what pecentage of male and female?

plt.figure(figsize = (16,9))
sns.countplot(y ='Branch', hue = "Gender", data = data) 
plt.xlabel('Count')
plt.ylabel('Product Type')
plt.show()

Normal customers vs Member?

plt.figure(figsize = (16,9))
sns.countplot(y ='Product line', hue = "Customer type", data = data) 
plt.xlabel('Count')
plt.ylabel('Product Type')
plt.show()

PAYMENT METHODS VS PRODUCTS?

plt.figure(figsize = (16,9))
sns.countplot(y ='Product line', hue = "Payment", data = data) 
plt.xlabel('Count')
plt.ylabel('Product Type')
plt.show()

Gender vs payment methods?

plt.figure(figsize = (16,9))
sns.countplot(y ='Gender', hue = "Payment", data = data) 
plt.xlabel('Count')
plt.ylabel('Product Type')
plt.show()

Branches vs Payment methods? Who gets more by card? who's getting more cash?

plt.figure(figsize = (16,9))
sns.countplot(y ='Branch', hue = "Payment", data = data) 
plt.xlabel('Count')
plt.ylabel('Product Type')
plt.show()

