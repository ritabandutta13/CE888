
# In[1]:


import pandas as pd
import numpy as np
import glob as glb
from datetime import date, timedelta


# - Collated n number of monthly stop and search data for all the constabularies which was available for on https://data.police.uk through a loop
# - Mapped constabularies from file names
# - Converted string date to month format
# - Imputed NaNs with 'Not Known' flag
# - Mapped self defined ethnicity to BAME, White & Not Known Buckets

# In[2]:


path = r'C:/Users/Ritaban Dutta/Downloads/s&s'
all_files = glb.glob(path + "\*-stop-and-search.csv")

li = []

for filename in all_files:

    df = pd.read_csv(filename, index_col=None, header=0,encoding='latin1')
    df['Place']=filename
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)
data=frame.copy(deep=True)
data.head()


# In[3]:


data=data.rename(columns={'Type':'n_type'})


# In[4]:


data['n_place']= data['Place'].str.split('\\').str[1]
data['n_place'] = data['n_place'].str[8:]
data['n_place']= data['n_place'].str.split('-stop-and-search.csv').str[0]

data['n_place']= data['n_place'].str.replace('-',' ')
data['n_place']=data['n_place'].str.title()

data.head()


# In[5]:


data['n_month']=data['Date'].str[0:7]


# In[6]:


data['n_date'] = data['Date']


# In[7]:


data['Part of a policing operation'].value_counts(dropna=False)
data['n_part_of_policing_operation']=np.where(data['Part of a policing operation']==True,'Part Of Policing Op','Not Known')
data['n_part_of_policing_operation']=np.where(data['Part of a policing operation']==False,'Not Part Of Policing Op',data['n_part_of_policing_operation'])
data['n_part_of_policing_operation'].value_counts(dropna=False)


# In[8]:


data['Outcome linked to object of search'].value_counts(dropna=False)
data['n_outcome_lnkd_to_obj_of_search']=np.where(data['Outcome linked to object of search']==True,'Obj Of Search Was Found','Not Known')
data['n_outcome_lnkd_to_obj_of_search']=np.where(data['Outcome linked to object of search']==False,'Obj Of Search Was Not Found',data['n_outcome_lnkd_to_obj_of_search'])
data['n_outcome_lnkd_to_obj_of_search'].value_counts(dropna=False)


# In[9]:


data['Gender'].value_counts(dropna=False)
data['n_gender']=np.where(data['Gender'].isna(),'Not Known',data['Gender'])
data['n_gender'].value_counts(dropna=False)


# In[10]:


data['Age range'].value_counts(dropna=False)
data['n_age']=np.where(data['Age range'].isna(),'Not Known',data['Age range'])
data['n_age']=np.where(data['Age range']=='over 34','Over 34',data['n_age'])
data['n_age']=np.where(data['Age range']=='under 10','Under 10',data['n_age'])
data['n_age'].value_counts(dropna=False)


# In[11]:


data['Self-defined ethnicity'].value_counts(dropna=False)


# In[12]:


data['n_self_def_eth']=np.where(data['Self-defined ethnicity'].isin(['White - English/Welsh/Scottish/Northern Irish/British',
                                                                     'White - Any other White background',
                                                                     'White - Gypsy or Irish Traveller',
                                                                     'White - Irish']),'White','Not Known')
data['n_self_def_eth']=np.where(data['Self-defined ethnicity'].isin(['Black/African/Caribbean/Black British - Any other Black/African/Caribbean background',
                                                                     'Black/African/Caribbean/Black British - African',
                                                                     'Black/African/Caribbean/Black British - Caribbean'
                                                                     ]),'BAME',data['n_self_def_eth'])
data['n_self_def_eth']=np.where(data['Self-defined ethnicity'].isin(['Asian/Asian British - Any other Asian background',
                                                                     'Asian/Asian British - Pakistani',
                                                                     'Asian/Asian British - Bangladeshi',
                                                                     'Asian/Asian British - Indian',
                                                                     'Asian/Asian British - Chinese']),'BAME',data['n_self_def_eth'])
data['n_self_def_eth']=np.where(data['Self-defined ethnicity'].isin(['Other ethnic group - Not stated',
                                                                     'Other ethnic group - Any other ethnic group',
                                                                     'Other ethnic group - Arab']),'BAME',data['n_self_def_eth'])
data['n_self_def_eth']=np.where(data['Self-defined ethnicity'].isin(['Mixed/Multiple ethnic groups - Any other Mixed/Multiple ethnic background',
                                                                     'Mixed/Multiple ethnic groups - White and Black Caribbean',
                                                                     'Mixed/Multiple ethnic groups - White and Black African',
                                                                     'Mixed/Multiple ethnic groups - White and Asian']),'BAME',data['n_self_def_eth'])


# In[13]:


data['n_self_def_eth'].value_counts()


# In[14]:


data['Officer-defined ethnicity'].value_counts(dropna=False)
data['n_off_def_eth']=np.where(data['Officer-defined ethnicity'].isna(),'Not Known',data['Officer-defined ethnicity'])
data['n_off_def_eth'].value_counts(dropna=False)


# In[15]:


data['Object of search'].value_counts(dropna=False)
data['n_obj_of_search']=np.where(data['Object of search'].isna(),'Not Known',data['Object of search'])
data['n_obj_of_search'].value_counts(dropna=False)


# In[16]:


data['Outcome'].value_counts()
data=data.rename(columns={'Outcome':'n_outcome'})
data['n_outcome'].value_counts()


# In[17]:


data['Removal of more than just outer clothing'].value_counts(dropna=False)
data['n_rem_more_than_out_cloth']=np.where(data['Removal of more than just outer clothing']==True,'Removed More Than Outer Clothing','Not Known')
data['n_rem_more_than_out_cloth']=np.where(data['Removal of more than just outer clothing']==False,'Not Removal Of More Than Outer Clothing',data['n_rem_more_than_out_cloth'])
data['n_rem_more_than_out_cloth'].value_counts(dropna=False)


# In[18]:


data=data.rename(columns={'Latitude':'n_latitude','Longitude':'n_longitude'})
data['n_month']=pd.to_datetime(data['n_month'],format="%Y-%m")
data['n_month']=data['n_month'].dt.strftime("%Y-%m")


# In[19]:


data=data.filter(regex='n_',axis=1)
data.head()


# In[24]:


data.to_csv(r'C:\Users\Ritaban Dutta\OneDrive\Desktop\PoliceBias.csv',index = False)


# ## EDA

# In[33]:


r0=data.groupby(['n_month','n_self_def_eth'])['n_type'].count().reset_index()
r01=data.groupby(['n_month'])['n_type'].count().reset_index()

r01=r01.rename(columns={'n_type':'n_type_tot'})
r0=r0.merge(r01,on='n_month',how='left')
r0['n_type']=round(r0['n_type']/r0['n_type_tot'],4)
r0=r0.drop(['n_type_tot'],axis=1)
r0B=r0.loc[r0.n_self_def_eth=='BAME']
r0B=r0B.rename(columns={'n_type':'%BAME'})

r0W=r0.loc[r0.n_self_def_eth=='White']
r0W=r0W.rename(columns={'n_type':'%White'})

r0W.head()
r0B.head()


# In[43]:


import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(20,8))
plt.title("MoM % Stop & Search Involving BAME", fontsize = 20)
blue =sns.lineplot(x = r0B['n_month'], y = r0B['%BAME'], data=r0B,markers=True)
red =sns.lineplot(x = r0W['n_month'], y = r0W['%White'], data=r0W,markers=True)


# ### In case of stop & search involving BAME 82% S&Ss are not part of police operations

# In[22]:


a=data.groupby(['n_self_def_eth','n_part_of_policing_operation'])['n_type'].count().reset_index()
a1=data.groupby(['n_self_def_eth'])['n_type'].count().reset_index()
a1=a1.rename(columns={'n_type':'n_type_tot'})
a=a.merge(a1,on=['n_self_def_eth'],how='left')
a['%perc_s&s']=round(a['n_type']/a['n_type_tot'],4)
a=a.drop(['n_type','n_type_tot'],axis=1)
a


# In[23]:


a=a.rename(columns={'%perc_s&s':''})
a=a.pivot_table(index=['n_self_def_eth'],
                      columns=['n_part_of_policing_operation'], aggfunc='mean').reset_index()
a.columns = a.columns.to_series().str.join('')

g=a.plot( 
  x = 'n_self_def_eth',  
  kind = 'bar',  
  stacked = True,  
  title = 'Distribution of Stop & Searches',  
  mark_right = True) 


# ### In case of stop & search involving BAME which are not part of police operations, times when object of search was found is significantly lesser

# In[24]:


a=data.loc[data.n_part_of_policing_operation=='Not Part Of Policing Op'].groupby(['n_self_def_eth','n_outcome_lnkd_to_obj_of_search'])['n_type'].count().reset_index()
a1=data.loc[data.n_part_of_policing_operation=='Not Part Of Policing Op'].groupby(['n_self_def_eth'])['n_type'].count().reset_index()


a1=a1.rename(columns={'n_type':'n_type_tot'})
a=a.merge(a1,on=['n_self_def_eth'],how='left')
a['%perc_s&s']=round(a['n_type']/a['n_type_tot'],4)
a=a.drop(['n_type','n_type_tot'],axis=1)


# In[25]:


a=a.rename(columns={'%perc_s&s':''})
a=a.pivot_table(index=['n_self_def_eth'],
                      columns=['n_outcome_lnkd_to_obj_of_search'], aggfunc='mean').reset_index()
a.columns = a.columns.to_series().str.join('')

g=a.plot( 
  x = 'n_self_def_eth',  
  kind = 'bar',  
  stacked = True,  
  title = 'Distribution of Stop & Searches Which Werent Part Of Policing Ops',  
  mark_right = True) 


# ### Cases were slightly more for BAME when no action was taken

# In[26]:


a=data.loc[data.n_part_of_policing_operation=='Not Part Of Policing Op'].groupby(['n_self_def_eth','n_outcome'])['n_type'].count().reset_index()
a1=data.loc[data.n_part_of_policing_operation=='Not Part Of Policing Op'].groupby(['n_self_def_eth'])['n_type'].count().reset_index()


a1=a1.rename(columns={'n_type':'n_type_tot'})
a=a.merge(a1,on=['n_self_def_eth'],how='left')
a['%perc_s&s']=round(a['n_type']/a['n_type_tot'],4)
a=a.drop(['n_type','n_type_tot'],axis=1)


# In[27]:


a=a.rename(columns={'%perc_s&s':''})
a=a.pivot_table(index=['n_self_def_eth'],
                      columns=['n_outcome'], aggfunc='mean').reset_index()
a.columns = a.columns.to_series().str.join('')

g=a.plot( 
  x = 'n_self_def_eth',  
  kind = 'bar',  
  stacked = True,  
  title = 'Distribution of Stop & Searches Which Werent Part Of Policing Ops',  
  mark_right = True) 


# In[45]:


r1=data.loc[(data.n_part_of_policing_operation=='Not Part Of Policing Op') &
           (data.n_outcome_lnkd_to_obj_of_search=='Obj Of Search Was Not Found')].groupby(['n_place','n_self_def_eth'])['n_type'].count().reset_index()
r11=data.loc[(data.n_part_of_policing_operation=='Not Part Of Policing Op') &
            (data.n_outcome_lnkd_to_obj_of_search=='Obj Of Search Was Not Found')].groupby(['n_place'])['n_type'].count().reset_index()


r11=r11.rename(columns={'n_type':'n_type_tot'})
r1=r1.merge(r11,on='n_place',how='left')
r1['n_type']=round(r1['n_type']/r1['n_type_tot'],4)
r1=r1.drop(['n_type_tot'],axis=1)
r1=r1.rename(columns={'n_type':''})
r1=r1.pivot_table(index=['n_place'],
                      columns=['n_self_def_eth'], aggfunc='mean').reset_index()
r1.columns = r1.columns.to_series().str.join('')

r1=r1.set_index('n_place')
r1.index.name = None
r1=r1.loc[:,['BAME']]
r1.head(100)


# In[29]:


from matplotlib import pyplot as plt
from matplotlib.ticker import StrMethodFormatter

ax = r1.plot(kind='barh', figsize=(8, 20), zorder=2, width=0.85)

# Despine
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Switch off ticks
ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")

# Draw vertical axis lines
vals = ax.get_xticks()
for tick in vals:
    ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

# Set x-axis label
ax.set_xlabel("Stop & Search Freq", labelpad=20, weight='bold', size=12)

# Set y-axis label
ax.set_ylabel("Constabulary", labelpad=20, weight='bold', size=12)

# Format y-axis label
ax.xaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))


# ## PREDICTIVE POLICING

# In[25]:


data=pd.read_csv(r'C:\Users\Ritaban Dutta\OneDrive\Desktop\PoliceBias.csv')
data=data.loc[#(data.n_self_def_eth!='Not Known') &
              #(data.n_month.isin(['2020-12'])) & 
               (data.n_month<='2018-04') &
#               (data.n_month<='2020-05') & 
              #(data.n_gender.isin(['Male','Female'])) &
              (data.n_age!='Not Known') #&
              #(data.n_outcome.isin(['A no further action disposal','Arrest']))
             ]

data.n_self_def_eth.value_counts()


# In[26]:


data.groupby(['n_month','n_self_def_eth'])['n_type'].count().reset_index().to_clipboard(index=False)


# In[27]:


data['dep_was_arrested_flag']=np.where(data['n_outcome']=='Arrest',1,0)
data.dep_was_arrested_flag.value_counts(dropna=False)
data=data.drop(['n_outcome','n_latitude','n_longitude'],axis=1)
data.head()


# In[28]:


data['n_bame_flag']=np.where(data['n_self_def_eth']=='BAME',1,0)


# In[29]:


df=data.drop(['n_date','n_self_def_eth','n_off_def_eth','n_rem_more_than_out_cloth','n_month'],axis=1)


# ### Encode categorical variables

# In[30]:


# Replace the categorical values with the numeric equivalents that we have above
categoricalFeatures = ['n_type', 'n_place',
                       'n_part_of_policing_operation', 'n_outcome_lnkd_to_obj_of_search',
                       'n_gender', 'n_age',
                       'n_obj_of_search']
# Iterate through the list of categorical features and one hot encode them.
for feature in categoricalFeatures:
    onehot = pd.get_dummies(df[feature], prefix=feature)
    df = df.drop(feature, axis=1)
    df = df.join(onehot)
df


# In[31]:


df.columns


# ### Separate dataset by x and y

# In[32]:


y = df['dep_was_arrested_flag']


# ### Create Test and Train splits

# In[33]:


from sklearn.model_selection import train_test_split
encoded_df = df.copy()
x = df.drop(['dep_was_arrested_flag'], axis = 1)


# In[34]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_std = scaler.fit_transform(x)
# We will follow an 80-20 split pattern for our training and test data, respectively
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.2, random_state = 0)


# ### Calculating actual disparate impact on testing values from original dataset
# Disparate Impact is defined as the ratio of favorable outcomes for the unprivileged group divided by the ratio of favorable outcomes for the priviliged group.
# The acceptable threshold is between .8 and 1.25, with .8 favoring the privileged group, and 1.25 favoring the unprivileged group.

# In[35]:


actual_test = x_test.copy()
actual_test['dep_was_arrested_flag'] = y_test
actual_test.shape


# In[36]:


bame_df = actual_test[actual_test['n_bame_flag'] == 1]
num_of_priviliged = bame_df.shape[0]
white_df = actual_test[actual_test['n_bame_flag'] == 0]
num_of_unpriviliged = white_df.shape[0]


# In[37]:


unpriviliged_outcomes = white_df[white_df['dep_was_arrested_flag'] == 1].shape[0]
unpriviliged_ratio = unpriviliged_outcomes/num_of_unpriviliged
unpriviliged_ratio


# In[38]:


priviliged_outcomes = bame_df[bame_df['dep_was_arrested_flag'] == 1].shape[0]
priviliged_ratio = priviliged_outcomes/num_of_priviliged
priviliged_ratio


# In[39]:


# Calculating disparate impact
disparate_impact = unpriviliged_ratio / priviliged_ratio
print("Disparate Impact, Ethnicity vs. Predicted Arrest Status: " + str(disparate_impact))


# ### Training a model on the original dataset

# In[40]:


from sklearn.linear_model import LogisticRegression
# Liblinear is a solver that is very fast for small datasets, like ours
model = LogisticRegression(solver='liblinear', class_weight='balanced')


# In[41]:


model.fit(x_train, y_train.astype('int'))


# ### Evaluating performance

# In[42]:


# Let's see how well it predicted with a couple values 
y_pred = pd.Series(model.predict(x_test))
y_test = y_test.reset_index(drop=True)
z = pd.concat([y_test, y_pred], axis=1)
z.columns = ['True', 'Prediction']
z.head()
# Predicts 4/5 correctly in this sample


# In[43]:


import matplotlib.pyplot as plt
from sklearn import metrics
print("Accuracy:", metrics.accuracy_score(y_test.astype('int'), y_pred.astype('int')))
print("Precision:", metrics.precision_score(y_test.astype('int'), y_pred.astype('int')))
print("Recall:", metrics.recall_score(y_test.astype('int'), y_pred.astype('int')))


# ### Calculating disparate impact on predicted values by model trained on original dataset

# In[44]:


# We now need to add this array into x_test as a column for when we calculate the fairness metrics.
y_pred = model.predict(x_test)
x_test['dep_was_arrested_flag'] = y_pred
original_output = x_test
original_output


# In[45]:


bame_df = original_output[original_output['n_bame_flag'] == 1]
num_of_priviliged = bame_df.shape[0]
white_df = original_output[original_output['n_bame_flag'] == 0]
num_of_unpriviliged = white_df.shape[0]


# In[46]:


unpriviliged_outcomes = white_df[white_df['dep_was_arrested_flag'] == 1].shape[0]
unpriviliged_ratio = unpriviliged_outcomes/num_of_unpriviliged
unpriviliged_ratio


# In[47]:


priviliged_outcomes = bame_df[bame_df['dep_was_arrested_flag'] == 1].shape[0]
priviliged_ratio = priviliged_outcomes/num_of_priviliged
priviliged_ratio


# In[48]:


disparate_impact = unpriviliged_ratio / priviliged_ratio
print("Disparate Impact, Ethnicity vs. Predicted Arrest Status: " + str(disparate_impact))


# ### Applying the Disparate Impact Remover to the dataset

# In[49]:


get_ipython().system('pip install tensorflow')


# In[56]:


import aif360


# In[57]:


from aif360.algorithms.preprocessing import DisparateImpactRemover


# In[58]:


get_ipython().system('pip install BlackBoxAuditing')


# In[59]:


# Must be a binaryLabelDataset
binaryLabelDataset = aif360.datasets.BinaryLabelDataset(
    favorable_label=1,
    unfavorable_label=0,
    df=encoded_df,
    label_names=['dep_was_arrested_flag'],
    protected_attribute_names=['n_bame_flag'])
di = DisparateImpactRemover(repair_level = 1.0)
dataset_transf_train = di.fit_transform(binaryLabelDataset)
transformed = dataset_transf_train.convert_to_dataframe()[0]
transformed


# ### Train a model using the dataset that underwent the pre-processing

# In[60]:


x_trans = transformed.drop(['dep_was_arrested_flag'], axis = 1)
y = transformed['dep_was_arrested_flag']
# Liblinear is a solver that is effective for relatively smaller datasets.
model = LogisticRegression(solver='liblinear', class_weight='balanced')
scaler = StandardScaler()
data_std = scaler.fit_transform(x_trans)
# Splitting into test and training
# We will follow an 80-20 split pattern for our training and test data
x_trans_train,x_trans_test,y_trans_train,y_trans_test = train_test_split(x_trans, y, test_size=0.2, random_state = 0)


# In[61]:


model.fit(x_trans_train, y_trans_train.astype('int'))


# ### Evaluating Performance

# In[62]:


# See how well it predicted with a couple values
y_trans_pred = pd.Series(model.predict(x_trans_test))
y_trans_test = y_trans_test.reset_index(drop=True)
z = pd.concat([y_trans_test, y_trans_pred], axis=1)
z.columns = ['True', 'Prediction']
z.head()
# Again, it predicts 4/5 correctly in this sample


# In[63]:


print("Accuracy:", metrics.accuracy_score(y_test.astype('int'), y_trans_pred.astype('int')))
print("Precision:", metrics.precision_score(y_test.astype('int'), y_trans_pred.astype('int')))
print("Recall:", metrics.recall_score(y_test.astype('int'), y_trans_pred.astype('int')))


# ### Calculating disparate impact on predicted values by model trained on transformed dataset

# In[64]:


# We now need to add this array into x_test as a column for when we calculate the fairness metrics.
y_trans_pred = model.predict(x_trans_test)
x_trans_test['dep_was_arrested_flag'] = y_trans_pred
transformed_output = x_trans_test
transformed_output


# In[65]:


bame_df = transformed_output[transformed_output['n_bame_flag'] == 1]
num_of_priviliged = bame_df.shape[0]
white_df = transformed_output[transformed_output['n_bame_flag'] == 0]
num_of_unpriviliged = white_df.shape[0]


# In[66]:


unpriviliged_outcomes = white_df[white_df['dep_was_arrested_flag'] == 1].shape[0]
unpriviliged_ratio = unpriviliged_outcomes/num_of_unpriviliged
unpriviliged_ratio


# In[67]:


priviliged_outcomes = bame_df[bame_df['dep_was_arrested_flag'] == 1].shape[0]
priviliged_ratio = priviliged_outcomes/num_of_priviliged
priviliged_ratio


# In[68]:


# Calculating disparate impact
disparate_impact = unpriviliged_ratio / priviliged_ratio
print("Disparate Impact, Ethnicity vs. Predicted Arrest Status: " + str(disparate_impact))

