#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tabulate
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('PS_20174392719_1491204439457_log.csv')
df


# In[3]:


print("Shape - (Rows, Columns):\n", df.shape)


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df.isna().sum() #No Null Values


# In[7]:


df.duplicated(keep='first').any() #No Duplicated Values


# In[8]:


summary_stats = df.groupby('type').agg({'type': 'count', 'amount': ['mean', 'median', lambda x: x.mode()[0]]})
summary_stats.columns = ['Count', 'Mean', 'Median', 'Mode']
fraud_stats = df.groupby(['type', 'isFraud']).size().unstack(fill_value=0)
fraud_stats.columns = ['Non-Fraud', 'Fraud']
result = pd.concat([summary_stats, fraud_stats], axis=1)
result


# In[9]:


df['nameOrig'].unique()


# In[10]:


df['type'].unique() # The different types of transactions of the TYPE field


# In[11]:


print("Steps - from {} to {}.".format(df['step'].min(), df['step'].max()))


# In[12]:


rate = df['isFraud'].value_counts()
rate


# In[13]:


ratio = rate/len(df.index)
print(f'The Ratio of fraudulent cases is:{ratio[1]}\nThe Ratio of non-fraudulent is:{ratio[0]}')


# In[14]:


pd.concat([df['isFraud'].value_counts(), 
                    (df['isFraud'].value_counts(normalize=True) * 100)], 
                    axis=1, 
                    keys=['Count', 'Percentage'])


# In[15]:


df['step_day_hour'] = (df['step']) % 24 #CONVERTING STEPS TO HOUR OF THE DAY
df


# In[16]:


df['step_day_week'] = (df['step']) % 7 #CONVERTING STEPS TO DAY OF THE WEEK
df


# In[17]:


plt.figure(figsize=(18,6))
plt.ylim(0, 10000000)
plt.title('Hourly Transaction Amounts')
ax = sns.scatterplot(x="step", y="amount", hue="isFraud",
                     data=df)


# In[18]:


df['step_day_month'] = (df['step']) // 24 #CONVERTING STEPS TO DAY OF THE MONTH
df


# In[19]:


df_fraud = df.loc[df.isFraud == 1] 
df_non_fraud = df.loc[df.isFraud == 0]


# In[20]:


df_fraud


# In[21]:


# fraud transactions amount value counts
df_fraud.amount.value_counts()


# In[22]:


#checking type of fraud transactions
df_fraud.type.value_counts()


# In[23]:


#PAYMENT
df_Fraud_Payment = df.loc[(df.isFraud == 1) & (df.type == 'PAYMENT')]
df_Total_Payment = df.loc[(df.type == 'PAYMENT')]

print("Number of fraudulent PAYMENTs = ", len(df_Fraud_Payment))
print("Number of non-fraudulent PAYMENTs = ", len(df.loc[(df.isFraud == 0) & (df.type == 'PAYMENT')]))
print("Fraud percentage: ", round(((len(df_Fraud_Payment) / len(df_Total_Payment)) * 100), 3), "%")


# In[24]:


#TRANSFER
df_Fraud_Transfer = df.loc[(df.isFraud == 1) & (df.type == 'TRANSFER')]
df_Total_Transfer = df.loc[(df.type == 'TRANSFER')]

print("Number of fraudulent TRANSFERs = ", len(df_Fraud_Transfer))
print("Number of non-fraudulent TRANSFERs = ", len(df.loc[(df.isFraud == 0) & (df.type == 'TRANSFER')]))
print("Fraud percentage: ", round(((len(df_Fraud_Transfer) / len(df_Total_Transfer)) * 100), 3), "%")


# In[25]:


#CASH_OUT
df_Fraud_CashOut = df.loc[(df.isFraud == 1) & (df.type == 'CASH_OUT')]
df_Total_CashOut = df.loc[(df.type == 'CASH_OUT')]

print("Number of fraudulent CASH_OUTs = ", len(df_Fraud_CashOut))
print("Number of non-fraudulent CASH_OUTs = ", len(df.loc[(df.isFraud == 0) & (df.type == 'CASH_OUT')]))
print("Fraud percentage: ", round(((len(df_Fraud_CashOut) / len(df_Total_CashOut)) * 100), 3), "%")


# In[26]:


#DEBIT
df_Fraud_Debit = df.loc[(df.isFraud == 1) & (df.type == 'DEBIT')]
df_Total_Debit = df.loc[(df.type == 'DEBIT')]

print("Number of fraudulent DEBITs = ", len(df_Fraud_Debit))
print("Number of non-fraudulent DEBITs = ", len(df.loc[(df.isFraud == 0) & (df.type == 'DEBIT')]))
print("Fraud percentage: ", round(((len(df_Fraud_Debit) / len(df_Total_Debit)) * 100), 3), "%")


# In[27]:


#CASH_IN
df_Fraud_CashIn = df.loc[(df.isFraud == 1) & (df.type == 'CASH_IN')]
df_Total_CashIn = df.loc[(df.type == 'CASH_IN')]

print("Number of fraudulent CASH_INs = ", len(df_Fraud_CashIn))
print("Number of fraudulent CASH_INs = ", len(df.loc[(df.isFraud == 0) & (df.type == 'CASH_IN')]))
print("Fraud percentage: ", round(((len(df_Fraud_CashIn) / len(df_Total_CashIn)) * 100), 3), "%")


# In[28]:


pd.concat([df_fraud.groupby('type')['amount'].mean(),df_non_fraud.groupby('type')['amount'].mean(),\
           df.groupby('type')['isFraud'].mean()*100],keys=["Fraudulent","Non-Fraudulent","Percent(%)"],axis=1,\
          sort=False).sort_values(by=['Non-Fraudulent'])


# In[29]:


fraudTrans = (len(df['isFraud'].loc[(df.isFraud == 1)]) / df.shape[0]) * 100
print("Total fraud transactions: {}%".format(round(fraudTrans, 2)))

total = [len(df_Total_Payment), len(df_Total_Transfer), len(df_Total_CashOut), len(df_Total_Debit), len(df_Total_CashIn)]
fraud = [len(df_Fraud_Payment), len(df_Fraud_Transfer), len(df_Fraud_CashOut), len(df_Fraud_Debit), len(df_Fraud_CashIn)]
names = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]

fig, axs = plt.subplots(1, 2, figsize=(15, 5), )

axs[0].bar(names, fraud)
axs[0].set_title("Fraud occurrences")
axs[0].set_ylabel("Transactions")

axs[1].bar(names, total)
axs[1].set_title("Total number of transactions")
axs[1].set_ylabel("Transactions")

plt.show()


# In[30]:


print("Transfers where isFraud predicts fraud",len(df.loc[df.isFraud==1]))
print("Tranfers where isFlaggedFraud prdicts fraud",len(df.loc[df.isFlaggedFraud==1]))


# In[31]:


df_Flagged = df.loc[df.isFlaggedFraud == 1]
df_Flagged

#In this table, the oldbalanceDest and newbalanceDest are identical (0.00).
# This might be because the transaction is halted when the threshold is reached. However isFlaggedFraud can remain 0 in TRANSFERs where oldbalanceDest and newbalanceDest can both be 0.
#Hence these conditions do not determine the state of isFlaggedFraud.


# In[32]:


print('\nNumber of TRANSFERs where isFlaggedFraud = 0, oldbalanceDest = 0 and newbalanceDest = 0: \n{}'.\
format(len(df_Total_Transfer.loc[(df_Total_Transfer.isFlaggedFraud == 0) & \
(df_Total_Transfer.oldbalanceDest == 0) & (df_Total_Transfer.newbalanceDest == 0)]))) #Even if the isFlaggedFraud condition is met, in some cases the its value remains 0.


# In[33]:


df_fraud_not_empty = df_fraud[(df_fraud['amount'] - df_fraud['oldbalanceOrg']) != 0]
df_fraud_not_empty


# In[34]:


df_fraud_empty = df_fraud[(df_fraud['amount'] - df_fraud['oldbalanceOrg']) == 0]
df_fraud_empty


# In[35]:


Not_Emptied=len(df_fraud_not_empty) #number of fraud transactions where the account is not emptied
Emptied = len(df_fraud_empty) #number of fraud transactions where the account is emptied


# In[36]:


df_non_fraud[df_non_fraud['isFlaggedFraud'] != df_non_fraud['isFraud']]
# the flagged fraud has been succesfull detecting non fraud trancaction 
# since there's no false positive on flagged fraud


# In[37]:


16 / 8213


# In[38]:


## checking outliers in this dataset

check_outliers = df[['step', 'amount', 'oldbalanceOrg', 'newbalanceOrig','oldbalanceDest','newbalanceDest']]

x = 1


plt.figure(figsize = (16, 8))
for column in check_outliers.columns:
    plt.subplot(3,2,x)
    sns.boxplot(check_outliers[column])
    x+=1
    
plt.tight_layout()

# we can see that there is a lot of outliers here as a normal data we would expect from a bank 
# there's some low amount of transaction and there are some high amount 
# there's some approach we can do for the step we can turn it into day instead of hour
# or we could  turn step from hour of month to hour of day


# In[39]:


df_fraud.groupby('nameOrig')['isFraud'].sum().sort_values(ascending = False)[0:10]

# all the fraud senders are unique 
# there's no sender that's succesfully commited multiple fraud


# In[40]:


df_fraud.groupby('nameDest')['isFraud'].sum().sort_values(ascending = False)[0:20]

# however there are many destination that's been in a multiple fraud case here


# In[41]:


print("Fraud")
print(df_fraud['step'].value_counts().reset_index().rename(columns={'index':'step', 'Step':'count'}).sort_values(by='step').reset_index(drop=True))

print("\nNon Fraud")
print(df_non_fraud['step'].value_counts().reset_index().rename(columns={'index':'step', 'Step':'count'}).sort_values(by='step').reset_index(drop=True))


# In[42]:


# Calculating frequencies and percentages for fraud data
fraud_step_freq = df_fraud['step_day_hour'].value_counts().reset_index()
fraud_step_freq.columns = ['step_day_hour', 'count']
fraud_step_freq['percentage'] = (fraud_step_freq['count'] / len(df_fraud)) * 100

# Calculating frequencies and percentages for non-fraud data
non_fraud_step_freq = df_non_fraud['step_day_hour'].value_counts().reset_index()
non_fraud_step_freq.columns = ['step_day_hour', 'count']
non_fraud_step_freq['percentage'] = (non_fraud_step_freq['count'] / len(df_non_fraud)) * 100

# Display fraud data
print("Fraud")
c1=fraud_step_freq.sort_values(by='step_day_hour',ascending =False).reset_index(drop=True)
c1

# Display non-fraud data
print("\nNon Fraud")
c2=non_fraud_step_freq.sort_values(by='step_day_hour',ascending =False).reset_index(drop=True)
c2


# In[43]:


# Calculate frequencies for fraud data and sort in descending order
fraud_step_freq = df_fraud['step_day_hour'].value_counts().reset_index().sort_values(by='count', ascending=False)
fraud_step_freq.columns = ['step_day_hour', 'count']

# Calculate frequencies for non-fraud data and sort in descending order
non_fraud_step_freq = df_non_fraud['step_day_hour'].value_counts().reset_index().sort_values(by='count', ascending=False)
non_fraud_step_freq.columns = ['step_day_hour', 'count']

# Calculate percentages for fraud data
fraud_step_freq['percentage'] = (fraud_step_freq['count'] / len(df_fraud)) * 100

# Calculate percentages for non-fraud data
non_fraud_step_freq['percentage'] = (non_fraud_step_freq['count'] / len(df_non_fraud)) * 100

# Display fraud data
print("Fraud")
print(fraud_step_freq.reset_index(drop=True))

# Display non-fraud data
print("\nNon Fraud")
print(non_fraud_step_freq.reset_index(drop=True))


### we can see that almost all transaction are spread out evenly for fraud transaction 
### however for the non fraud transaction there are very low percentage of transaction before 9am
### the highest percentage of transaction for non fraud transaction happened during 18.00 - 20.00


# In[44]:


# Calculate frequencies and percentages for fraud data
fraud_step_freq = df_fraud['step_day_hour'].value_counts().reset_index()
fraud_step_freq.columns = ['step_day_hour', 'count']
fraud_step_freq['percentage'] = (fraud_step_freq['count'] / len(df_fraud)) * 100

# Calculate frequencies and percentages for non-fraud data
non_fraud_step_freq = df_non_fraud['step_day_hour'].value_counts().reset_index()
non_fraud_step_freq.columns = ['step_day_hour', 'count']
non_fraud_step_freq['percentage'] = (non_fraud_step_freq['count'] / len(df_non_fraud)) * 100

# Sort the dataframes by count in descending order
fraud_sorted = fraud_step_freq.sort_values(by='count', ascending=False).reset_index(drop=True)
non_fraud_sorted = non_fraud_step_freq.sort_values(by='count', ascending=False).reset_index(drop=True)

# Display fraud data as a table
print("Fraud:")
fraud_sorted

# Display non-fraud data as a table
print("\nNon Fraud:")
non_fraud_sorted


# In[45]:


# what time does the fraud transaction usually happened

df_fraud.groupby('type')['step_day_hour'].mean()

# both cashout and transfer fraud usually happened at 11am


# In[46]:


plt.figure(figsize=(15, 6))
plt.subplot(1, 2, 1)
sns.countplot(x=df_fraud['step_day_hour'], hue=df_fraud['step_day_hour'], legend=False)
plt.title('Fraud Transaction')

plt.subplot(1, 2, 2)
sns.countplot(x=df_non_fraud['step_day_hour'])
plt.title('Non Fraud Transaction')

plt.show()


# In[47]:


plt.figure(figsize=(10, 3))
sns.distplot(df_non_fraud.step, label="Genuine Transaction")
sns.distplot(df_fraud.step, label='Fraud Transaction')
plt.xlabel('Hour')
plt.ylabel('Number of Transactions')
plt.title('Distribution of Transactions over the Time')
plt.legend()


# In[61]:


#FOR step_day_week
# Calculate frequencies and percentages for fraud data based on step_day_week
fraud_step_week_freq = df_fraud['step_day_week'].value_counts().reset_index()
fraud_step_week_freq.columns = ['step_day_week', 'count']
fraud_step_week_freq['percentage'] = (fraud_step_week_freq['count'] / len(df_fraud)) * 100

# Calculate frequencies and percentages for non-fraud data based on step_day_week
non_fraud_step_week_freq = df_non_fraud['step_day_week'].value_counts().reset_index()
non_fraud_step_week_freq.columns = ['step_day_week', 'count']
non_fraud_step_week_freq['percentage'] = (non_fraud_step_week_freq['count'] / len(df_non_fraud)) * 100

# Sort the dataframes by count in descending order
fraud_sorted_week = fraud_step_week_freq.sort_values(by='count', ascending=False).reset_index(drop=True)
non_fraud_sorted_week = non_fraud_step_week_freq.sort_values(by='count', ascending=False).reset_index(drop=True)

# Display fraud data as a table
print("Fraud:")
fraud_sorted_week

# Display non-fraud data as a table
print("\nNon Fraud:")
non_fraud_sorted_week


# In[49]:


df_fraud.groupby('type')['step_day_week'].mean()


# In[58]:


plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.countplot(x=df_fraud['step_day_week'], palette='Set1')
plt.title('Fraud Transaction')

plt.subplot(1, 2, 2)
sns.countplot(x=df_non_fraud['step_day_week'], palette='Set2')
plt.title('Non Fraud Transaction')

plt.show()


# In[60]:


#FOR step_day_month
# Calculate frequencies and percentages for fraud data based on step_day_month
fraud_step_month_freq = df_fraud['step_day_month'].value_counts().reset_index()
fraud_step_month_freq.columns = ['step_day_month', 'count']
fraud_step_month_freq['percentage'] = (fraud_step_month_freq['count'] / len(df_fraud)) * 100

# Calculate frequencies and percentages for non-fraud data based on step_day_month
non_fraud_step_month_freq = df_non_fraud['step_day_month'].value_counts().reset_index()
non_fraud_step_month_freq.columns = ['step_day_month', 'count']
non_fraud_step_month_freq['percentage'] = (non_fraud_step_month_freq['count'] / len(df_non_fraud)) * 100

# Sort the dataframes by count in descending order
fraud_sorted_month = fraud_step_month_freq.sort_values(by='count', ascending=False).reset_index(drop=True)
non_fraud_sorted_month = non_fraud_step_month_freq.sort_values(by='count', ascending=False).reset_index(drop=True)

# Display fraud data as a table
print("Fraud:")
fraud_sorted_month

# Display non-fraud data as a table
print("\nNon Fraud:")
non_fraud_sorted_month


# In[52]:


df_fraud.groupby('type')['step_day_month'].mean()


# In[59]:


plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
sns.countplot(x=df_fraud['step_day_month'], palette='Set1')
plt.title('Fraud Transaction')

plt.subplot(1, 2, 2)
sns.countplot(x=df_non_fraud['step_day_month'], palette='Set2')
plt.title('Non Fraud Transaction')

plt.show()


# In[54]:


## median amount for each transaction type 

df_fraud.groupby('type')['amount'].median()

# we can see from the fraud  that the transfer type has almost equal median amount compared to 
# cashout type 


# In[55]:


plt.figure(figsize = (15, 6))
plt.subplot(1,2,1)
sns.countplot(df_fraud['type'])
plt.title('Number of  Fraud Transaction')
plt.subplot(1,2,2)
sns.countplot(df_non_fraud['type'])
plt.title('Number of Non Fraud Transaction')

plt.tight_layout()
plt.show()

# for the non fraud transaction there are 5 types of transaction 
# compared to the fraud transaction there are only 2 types of transaction
# cashout is the most popular payment in non fraud transaction payment comes 2nd
# while debit is the least popular transaction in non fraud transaction

# for the fraud transaction transfer and cashout are the only transaction
# both types of transaction have almost similar number of transaction


# In[56]:


plt.figure(figsize = (15, 6))
plt.subplot(1,2,1)
sns.barplot(df_fraud['type'], estimator = np.median )
plt.title('Median Fraud transaction for each type of transaction')
plt.subplot(1,2,2)
sns.barplot(df_non_fraud['type'], estimator = np.median)
plt.title('Median non Fraud Transaction for each transfer type')
# we can see that the median of cashout fraud transaction is similar to transfer fraud
# the median amount for non fraud transaction shows that transfer has the highest average among
# all transaction type

# the median of cashout fraud transaction is higher ccommpared to median non fraud transaction 

# there are only 2 types of transaction in fraud transaction (Cashout, transfer)
# there are 5 types of transaction in non fraud transaction


# In[ ]:




