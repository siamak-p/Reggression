
# coding: utf-8

# In[1]:


import os
import numpy as np
import matplotlib.pyplot as plt
import keras
import pandas as pd
from keras.layers import Dense, SimpleRNN
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.cross_validation import train_test_split
import keras.backend as K


# In[2]:


# load train and test data and drop index column
df_train = pd.read_csv('/home/siamak/Projects/deep_learning/src/test_program/train_data/data.csv', index_col=0)
df_test = pd.read_csv('/home/siamak/Projects/deep_learning/src/test_program/test_data/test.csv', index_col=0)


# In[3]:


print(df_train.shape)
print(df_test.shape)


# # Preprocessing

# In[4]:


# drop unnecessary columns in train and test data
df_train.drop(['Start_time', 'End_time', 'Name of show', 'Name of episode'], axis=1, inplace=True)
df_test.drop(['Start_time', 'End_time', 'Name of show', 'Name of episode'], axis=1, inplace=True)


# In[5]:


# get the train label
train_label = df_train['Market Share_total']
df_train.drop(['Market Share_total'], axis=1, inplace=True)


# In[6]:


print(df_train.head(1))
print('-------------------------------------------------')
print(df_test.head(1))


# In[7]:


# fill nan value with 0 in train and test data
df_train.fillna(0, inplace=True)
df_test.fillna(0, inplace=True)


# In[8]:


# convert nominal features to numerical for train and test data
lb_make = LabelEncoder()
df_train["Episode"] = lb_make.fit_transform(df_train["Episode"])
df_train["Station"] = lb_make.fit_transform(df_train["Station"])
df_train["Channel Type"] = lb_make.fit_transform(df_train["Channel Type"])
df_train["Season"] = lb_make.fit_transform(df_train["Season"])
df_train["Year"] = lb_make.fit_transform(df_train["Year"])
df_train["Date"] = lb_make.fit_transform(df_train["Date"])
df_train["Day of week"] = lb_make.fit_transform(df_train["Day of week"])
df_train["Genre"] = lb_make.fit_transform(df_train["Genre"])
df_train["First time or rerun"] = lb_make.fit_transform(df_train["First time or rerun"])
df_train["# of episode in the season"] = lb_make.fit_transform(df_train["# of episode in the season"])
df_train["Movie?"] = lb_make.fit_transform(df_train["Movie?"])
df_train["Game of the Canadiens during episode?"] = lb_make.fit_transform(df_train["Game of the Canadiens during episode?"])

# df_train["Start_time"] = lb_make.fit_transform(df_train["Start_time"])
# df_train["End_time"] = lb_make.fit_transform(df_train["End_time"])
# df_train["Name of show"] = lb_make.fit_transform(df_train["Name of show"])
# df_train["Name of episode"] = lb_make.fit_transform(df_train["Name of episode"])


df_test["Episode"] = lb_make.fit_transform(df_test["Episode"])
df_test["Station"] = lb_make.fit_transform(df_test["Station"])
df_test["Channel Type"] = lb_make.fit_transform(df_test["Channel Type"])
df_test["Season"] = lb_make.fit_transform(df_test["Season"])
df_test["Year"] = lb_make.fit_transform(df_test["Year"])
df_test["Date"] = lb_make.fit_transform(df_test["Date"])
df_test["Day of week"] = lb_make.fit_transform(df_test["Day of week"])
df_test["Genre"] = lb_make.fit_transform(df_test["Genre"])
df_test["First time or rerun"] = lb_make.fit_transform(df_test["First time or rerun"])
df_test["# of episode in the season"] = lb_make.fit_transform(df_test["# of episode in the season"])
df_test["Movie?"] = lb_make.fit_transform(df_test["Movie?"])
df_test["Game of the Canadiens during episode?"] = lb_make.fit_transform(df_test["Game of the Canadiens during episode?"])

# df_test["Start_time"] = lb_make.fit_transform(df_test["Start_time"])
# df_test["End_time"] = lb_make.fit_transform(df_test["End_time"])
# df_test["Name of show"] = lb_make.fit_transform(df_test["Name of show"])
# df_test["Name of episode"] = lb_make.fit_transform(df_test["Name of episode"])


# In[9]:


print(df_train.head(1))
print(df_test.head(1))


# In[10]:


print(df_train.shape)
print(df_test.shape)


# In[11]:


# Normalize with min_max normalizer
min_max_normalizer = MinMaxScaler()
min_max_normalizer.fit(df_train)
#normalize train data
train_data = min_max_normalizer.transform(df_train)

#normalize test data
test_data = min_max_normalizer.transform(df_test)


# In[12]:


print(type(train_data))


# In[13]:


print(train_data[1])
print(test_data[1])


# # Make model

# In[14]:


# dadehaye test chon label nadarand ta model ra behtar arzyabi konim, 20 hezar dade az train joda karde va serfan jahate test model az anha estefade shode
print(train_data.shape)
x_test = train_data[:20000]
y_test = train_label[:20000]
new_train_data = train_data[20000:]
new_train_label = train_label[20000:]
x_train, x_val, y_train, y_val = train_test_split(new_train_data, new_train_label, test_size=0.2, random_state=42)


# In[15]:


def initializer(weight_matrix):
    return K.random_uniform(shape=weight_matrix, minval=-1.2, maxval=0.8, seed=(142))


# In[ ]:


model = Sequential()
model.add(Dense(64, activation='relu',
                input_shape=(train_data.shape[1],), kernel_initializer=initializer, bias_initializer='zeros'))
model.add(Dense(32, activation='relu', kernel_initializer=initializer, bias_initializer='zeros'))
model.add(Dense(16, activation='relu', kernel_initializer=initializer, bias_initializer='zeros'))
model.add(Dense(8, activation='relu', kernel_initializer=initializer, bias_initializer='zeros'))
model.add(Dense(4, activation='relu', kernel_initializer=initializer, bias_initializer='zeros'))
model.add(Dense(1, kernel_initializer=initializer, bias_initializer='zeros'))
model.compile(optimizer='adam', loss='mae', metrics=['mae'])


# In[ ]:


history = model.fit(x_train, y_train, epochs=500, batch_size=512, validation_data=(x_val, y_val))


# In[ ]:


mae_history = history.history['mean_absolute_error']


# In[ ]:


plt.plot(range(1, 501), mae_history, 'b', label='mean_absolute_error')
plt.xlabel('epochs')
plt.ylabel('mae_validation')
plt.title('Mean_absolute_error validation')
plt.legend()
plt.show()


# In[ ]:


test_mse_score, test_mae_score = model.evaluate(x_test, y_test)
print(test_mae_score)


# In[ ]:


prd = model.predict(x_test)
print(prd[:40])
print(y_test[:40])


# In[27]:


model2=0
model2 = Sequential()
model2.add(Dense(64, activation='relu',
                input_shape=(train_data.shape[1],)))
model2.add(Dense(32, activation='relu'))
model2.add(Dense(16, activation='relu'))
model2.add(Dense(8, activation='relu'))
model2.add(Dense(4, activation='relu'))
model2.add(Dense(1))
model2.compile(optimizer='adam', loss='mae', metrics=['mae'])
model2.summary()
model2.fit(x_train, y_train, epochs=50, batch_size=512, validation_data=(x_val, y_val), shuffle=False)


# In[28]:


test_mse_score, test_mae_score = model2.evaluate(x_test, y_test)
print(test_mae_score)


# In[29]:


# chon test set label nadasht, 20000 record az train joda va baraye test avaliye kenar gozashte shod. in 20000 dade
# dar faze amoozesh hich tasiri nadashte
prd = model2.predict(x_test)
print(prd[:30])
print(y_test[:30])


# In[30]:


# dade haye test, tavasote modele train shode predict shode va natije dar zir aamade
prd_test = model2.predict(test_data)


# In[31]:


print(prd_test[:20])

