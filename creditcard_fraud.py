import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,ConfusionMatrixDisplay
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv('C:/Users/Prasanthi Rani/Downloads/creditcard.csv')
df.head()

df.shape
df.columns
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()
df.drop_duplicates(inplace=True)
df.shape
df['Class'].unique()
df['Class'].value_counts()

fraud = df[df['Class'] == 1]
normal = df[df['Class'] == 0]
normal_percentage = len(normal)/(len(fraud)+len(normal))
fraud_percentage = len(fraud)/(len(fraud)+len(normal))
print('Percentage of fraud transactions = ', round(fraud_percentage * 100, 3))
print('Percentage of normal transactions = ', round(normal_percentage * 100, 3))

plt.figure(figsize=(7,7))
sns.countplot(data=df,x='Class',palette=['skyblue','red'])
plt.title("Number of Normal and Fraud Transactions");

plt.figure(figsize=(8,6))
sns.FacetGrid(df, hue="Class", height=6,palette=['skyblue','red']).map(plt.scatter, "Time", "Amount").add_legend()
plt.show()

X = df.drop('Class',axis=1)
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def model_train_test(model,X_train,y_train,X_test,y_test):
    model.fit(X_train,y_train)
    prediction = model.predict(X_test)
    print('Accuracy = {}'.format(accuracy_score(y_test,prediction)))
    print(classification_report(y_test,prediction))
    matrix = confusion_matrix(y_test,prediction)
    dis = ConfusionMatrixDisplay(matrix)
    dis.plot()
    plt.show()

rf_model = RandomForestClassifier()
model_train_test(rf_model,X_train,y_train,X_test,y_test)

Decision_tree = DecisionTreeClassifier()
model_train_test(Decision_tree,X_train,y_train,X_test,y_test)