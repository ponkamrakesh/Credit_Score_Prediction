import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from imblearn.over_sampling import SMOTE

Print("Credit Score Prediction")

# Load the dataset
df_train_original = pd.read_csv(r"..\Desktop\Rakesh_DS\DataSet\Credit_Score_DataSet.csv",low_memory=False)
df_train = df_train_original.copy()
print(df_train.head())

def get_column_details(df, column):
    print(f"Details of '{column}' column:")
    print(f"DataType: {df[column].dtype}")
    count_null = df[column].isnull().sum()
    print(f"Null values: {count_null}")
    print(f"Number of Unique Values: {df[column].nunique()}")
    print(f"Distribution of column:\n{df[column].value_counts()}")

def fill_missing_with_group_mode(df, groupby, column):
    print(f"Missing values before filling: {df[column].isnull().sum()}")
    mode_per_group = df.groupby(groupby)[column].transform(lambda x: x.mode().iat[0])
    df[column] = df[column].fillna(mode_per_group)
    print(f"Missing values after filling: {df[column].isnull().sum()}")

def clean_categorical_field(df, groupby, column, replace_value=None):
    print(f"Cleaning '{column}' column")
    if replace_value is not None:
        df[column] = df[column].replace(replace_value, np.nan)
        print(f"Replaced '{replace_value}' with NaN")
    fill_missing_with_group_mode(df, groupby, column)

def fix_inconsistent_values(df, groupby, column):
    print(f"Existing Min, Max Values:\n{df[column].apply([min, max])}")
    df_dropped = df[df[column].notna()].groupby(groupby)[column].apply(list)
    x, y = df_dropped.apply(lambda x: stats.mode(x)).apply([min, max])
    mini, maxi = x[0], y[0]
    col = df[column].apply(lambda x: np.nan if ((x < mini) | (x > maxi) | (x < 0)) else x)
    mode_by_group = df.groupby(groupby)[column].transform(lambda x: x.mode()[0] if not x.mode().empty else np.NaN)
    df[column] = col.fillna(mode_by_group)
    df[column].fillna(df[column].mean(), inplace=True)
    print(f"After Cleaning Min, Max Values:\n{df[column].apply([min, max])}")
    print(f"Unique values after Cleaning: {df[column].nunique()}")
    print(f"Null values after Cleaning: {df[column].isnull().sum()}")

def clean_numerical_field(df, groupby, column, strip=None, datatype=None, replace_value=None):
    print(f"Cleaning '{column}' column")
    if replace_value is not None:
        df[column] = df[column].replace(replace_value, np.nan)
        print(f"Replaced '{replace_value}' with NaN")
    if df[column].dtype == object and strip is not None:
        df[column] = df[column].str.strip(strip)
        print(f"Removed trailing & leading '{strip}'")
    if datatype is not None:
        df[column] = df[column].astype(datatype)
        print(f"Changed datatype of '{column}' to {datatype}")
    fix_inconsistent_values(df, groupby, column)

def plot_countplot(df, column, user_friendly_column_name, rotation=0):
    print(f"Plotting distribution of '{user_friendly_column_name}'")
    sns.set_palette("deep")
    sns.countplot(data=df, x=column)
    plt.xlabel(user_friendly_column_name)
    plt.ylabel('Number of Records')
    plt.title(f'{user_friendly_column_name} Distribution')
    plt.xticks(rotation=rotation)
    plt.show()

def plot_displot(df, column, user_friendly_column_name, rotation=0, bins=20):
    print(f"Plotting distribution of '{user_friendly_column_name}'")
    sns.set_palette("deep")
    sns.displot(data=df, x=column, kde=True, bins=bins)
    plt.xlabel(user_friendly_column_name)
    plt.ylabel('Number of Records')
    plt.title(f'{user_friendly_column_name} Distribution')
    plt.xticks(rotation=rotation)
    plt.show()

def plot_stacked_bar(df, column1, column2, rotation=0):
    print(f"Plotting stacked bar for '{column1}' and '{column2}'")
    sns.set_palette("deep")
    pd.crosstab(df[column1], df[column2]).plot(kind='bar', stacked=True)
    plt.xlabel(column1)
    plt.ylabel('Number of Records')
    plt.title(f'{column1} & {column2} Distribution')
    plt.xticks(rotation=rotation)
    plt.show()

# Example usage of the functions
column_name = 'Month'
get_column_details(df_train, column_name)
plot_stacked_bar(df_train, column_name, 'Credit_Score')
df_train['Month'] = pd.to_datetime(df_train.Month, format='%B').dt.month

column_name = 'Name'
group_by = 'Customer_ID'
get_column_details(df_train, column_name)
clean_categorical_field(df_train, group_by, column_name)

column_name = 'SSN'
group_by = 'Customer_ID'
garbage_value = '#F%$D@*&8'
get_column_details(df_train, column_name)
clean_categorical_field(df_train, group_by, column_name, garbage_value)

column_name = 'Occupation'
group_by = 'Customer_ID'
garbage_value = '_______'
user_friendly_name = 'Occupation'
get_column_details(df_train, column_name)
clean_categorical_field(df_train, group_by, column_name, garbage_value)
plot_stacked_bar(df_train, column_name, 'Credit_Score', rotation=60)

df_train['Type_of_Loan'].replace([np.nan], 'Not Specified', inplace=True)
column_name = 'Credit_Mix'
group_by = 'Customer_ID'
garbage_value = '_'
get_column_details(df_train, column_name)
clean_categorical_field(df_train, group_by, column_name, garbage_value)
plot_stacked_bar(df_train, column_name, 'Credit_Score', rotation=60)

column_name = 'Payment_of_Min_Amount'
get_column_details(df_train, column_name)
plot_stacked_bar(df_train, column_name, 'Credit_Score', rotation=60)

column_name = 'Payment_Behaviour'
group_by = 'Customer_ID'
garbage_value = '!@9#%8'
get_column_details(df_train, column_name)
clean_categorical_field(df_train, group_by, column_name, garbage_value)
plot_stacked_bar(df_train, column_name, 'Credit_Score', rotation=80)

column_name = 'Age'
group_by = 'Customer_ID'
user_friendly_name = 'Age'
get_column_details(df_train, column_name)
clean_numerical_field(df_train, group_by, column_name, strip='_', datatype='int')
plot_displot(df_train, column_name, user_friendly_name, bins=40)

column_name = 'Annual_Income'
group_by = 'Customer_ID'
user_friendly_name = 'Annual Income'
get_column_details(df_train, column_name)
clean_numerical_field(df_train, group_by, column_name, strip='_', datatype='float')
plot_displot(df_train, column_name, user_friendly_name, bins=40)

column_name = 'Monthly_Inhand_Salary'
group_by = 'Customer_ID'
user_friendly_name = 'Monthly Inhand Salary'
get_column_details(df_train, column_name)
clean_numerical_field(df_train, group_by, column_name)
plot_displot(df_train, column_name, user_friendly_name, bins=40)

column_name = 'Num_Bank_Accounts'
group_by = 'Customer_ID'
user_friendly_name = 'Number of Bank Accounts'
get_column_details(df_train, column_name)
clean_numerical_field(df_train, group_by, column_name)
plot_countplot(df_train, column_name, user_friendly_name)

column_name = 'Num_Credit_Card'
group_by = 'Customer_ID'
user_friendly_name = 'Number of Credit Card'
get_column_details(df_train, column_name)
clean_numerical_field(df_train, group_by, column_name)
plot_countplot(df_train, column_name, user_friendly_name)

column_name = 'Interest_Rate'
group_by = 'Customer_ID'
user_friendly_name = 'Interest Rate'
get_column_details(df_train, column_name)
clean_numerical_field(df_train, group_by, column_name)
plot_countplot(df_train, column_name, user_friendly_name, rotation=90)

column_name = 'Delay_from_due_date'
group_by = 'Customer_ID'
user_friendly_name = 'Delay from Due Date'
get_column_details(df_train, column_name)
clean_numerical_field(df_train, group_by, column_name)
plot_displot(df_train, column_name, user_friendly_name, rotation=90)

column_name = 'Num_of_Delayed_Payment'
group_by = 'Customer_ID'
user_friendly_name = 'Number of Delayed Payment'
get_column_details(df_train, column_name)
clean_numerical_field(df_train, group_by, column_name, strip='_', datatype='float')
plot_countplot(df_train, column_name, user_friendly_name, rotation=90)

column_name = 'Changed_Credit_Limit'
group_by = 'Customer_ID'
user_friendly_name = 'Changed Credit Limit'
get_column_details(df_train, column_name)
clean_numerical_field(df_train, group_by, column_name, strip='_', datatype='float', replace_value='_')
plot_displot(df_train, column_name, user_friendly_name, rotation=90)

column_name = 'Num_Credit_Inquiries'
group_by = 'Customer_ID'
user_friendly_name = 'Number of Credit Inquiries'
get_column_details(df_train, column_name)
clean_numerical_field(df_train, group_by, column_name)
plot_countplot(df_train, column_name, user_friendly_name, rotation=90)

column_name = 'Outstanding_Debt'
group_by = 'Customer_ID'
user_friendly_name = 'Outstanding Debt'
get_column_details(df_train, column_name)
clean_numerical_field(df_train, group_by, column_name, strip='_', datatype=float)
plot_displot(df_train, column_name, user_friendly_name, rotation=90)

column_name = 'Credit_Utilization_Ratio'
group_by = 'Customer_ID'
user_friendly_name = 'Credit Utilization Ratio'
get_column_details(df_train, column_name)
plot_displot(df_train, column_name, user_friendly_name)

def Month_Converter(val):
    if pd.notnull(val):
        years = int(val.split(' ')[0])
        month = int(val.split(' ')[3])
        return (years * 12) + month
    else:
        return val

df_train['Credit_History_Age'] = df_train['Credit_History_Age'].apply(lambda x: Month_Converter(x)).astype(float)
column_name = 'Credit_History_Age'
group_by = 'Customer_ID'
user_friendly_name = 'Credit History Age'
get_column_details(df_train, column_name)
clean_numerical_field(df_train, group_by, column_name, datatype=float)
plot_displot(df_train, column_name, user_friendly_name)

column_name = 'Total_EMI_per_month'
group_by = 'Customer_ID'
user_friendly_name = 'Total EMI per month'
get_column_details(df_train, column_name)
clean_numerical_field(df_train, group_by, column_name)
plot_displot(df_train, column_name, user_friendly_name)

column_name = 'Amount_invested_monthly'
group_by = 'Customer_ID'
user_friendly_name = 'Amount invested monthly'
get_column_details(df_train, column_name)
clean_numerical_field(df_train, group_by, column_name, datatype=float, strip='_')
plot_displot(df_train, column_name, user_friendly_name, bins=100)

column_name = 'Monthly_Balance'
group_by = 'Customer_ID'
user_friendly_name = 'Monthly Balance'
get_column_details(df_train, column_name)
df_train[column_name].replace('', np.nan)
clean_numerical_field(df_train, group_by, column_name, strip='_', datatype=float, replace_value='__-333333333333333333333333333__')
plot_displot(df_train, column_name, user_friendly_name, bins=30)

column_name = 'Num_of_Loan'
group_by = 'Customer_ID'
user_friendly_name = 'Number of Loan'
get_column_details(df_train, column_name)
clean_numerical_field(df_train, group_by, column_name, strip='_', datatype=float)
plot_displot(df_train, column_name, user_friendly_name, bins=30)

# Drop unnecessary columns
print(f"Dataset size before dropping columns: {df_train.shape}")
drop_columns = ['ID', 'Customer_ID', 'Name', 'SSN']
df_train.drop(drop_columns, axis=1, inplace=True)
print(f"Dataset size after dropping columns: {df_train.shape}")

# Encode categorical columns
categorical_columns = ['Occupation', 'Type_of_Loan', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour', 'Credit_Score']
label_encoder = LabelEncoder()
for column in categorical_columns:
    df_train[column] = label_encoder.fit_transform(df_train[column])

# Split data into features and target
X = df_train.drop('Credit_Score', axis=1)
y = df_train['Credit_Score']
print(X.shape)
print(y.shape)

# Normalize data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=17, stratify=y)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

def evaluate_model(y_test, y_pred):
    print("Classification Report")
    print(classification_report(y_test, y_pred))
    print("\nConfusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, cmap='Greens', fmt='.0f')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.show()

# List of classifiers to test
classifiers = [
    ('Decision Tree', DecisionTreeClassifier()),
    ('Random Forest', RandomForestClassifier()),
    ('KNN', KNeighborsClassifier(n_neighbors=5)),
    ('Gaussian NB', GaussianNB()),
    ('XGB', xgb.XGBClassifier())
]

# Evaluate each classifier
for clf_name, clf in classifiers:
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    avg_accuracy = scores.mean()
    avg_precision = cross_val_score(clf, X_train, y_train, cv=5, scoring='precision_macro').mean()
    avg_recall = cross_val_score(clf, X_train, y_train, cv=5, scoring='recall_macro').mean()
    print(f'Classifier: {clf_name}')
    print(f'Average Accuracy: {avg_accuracy:.4f}')
    print(f'Average Precision: {avg_precision:.4f}')
    print(f'Average Recall: {avg_recall:.4f}')
    print('-----------------------')

# Train and evaluate Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
evaluate_model(y_test, y_pred)

# Handle imbalanced data using SMOTE
smote = SMOTE()
X_sm, y_sm = smote.fit_resample(X, y)
print(y_sm.value_counts())

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_sm, y_sm, test_size=0.2, random_state=15, stratify=y_sm)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Train and evaluate Random Forest classifier on balanced data
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
evaluate_model(y_test, y_pred)
