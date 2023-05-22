#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow
print(tensorflow.__version__)


# In[2]:


from tensorflow.keras import Sequential
from keras.preprocessing import sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  SimpleRNN, Embedding,BatchNormalization, LSTM
from tensorflow.keras.layers import Dense, Activation, Input, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd    


# # Data Checking

# In[3]:


df = pd.read_csv("seq_train.csv",  header=None,skiprows=1)


# In[4]:


print(df)


# In[5]:


print(df[1])
print(df[1].shape)


# In[6]:


df_check=pd.read_csv('seq_train.csv', header=None,nrows=5000,skiprows=1)


# In[7]:


print(df_check)
print(df_check.shape)


# In[8]:


df_check=df_check[1].str.split(' ',expand=True)


# In[9]:


print(df_check)


# In[10]:


check_y= df_check.iloc[: , 256:]
check_x= df_check.iloc[: , :256]


# In[11]:


print(check_y)


# In[12]:


print(check_x)


# # Model 

# In[13]:


seqnc_lngth = 256
vocab_size = 3537 
embddng_dim = 100

inpt_vec = Input(shape=(seqnc_lngth,))
l1 = Embedding(vocab_size, embddng_dim, input_length=seqnc_lngth)(inpt_vec)
l2 = Dropout(0.3)(l1)
l3 = LSTM(100, activation='tanh',recurrent_activation='sigmoid')(l2)
l4 = BatchNormalization()(l3)
l5 = Dropout(0.3)(l4)
l6 = Dense(vocab_size, activation='softmax')(l5)
lstm = Model(inpt_vec, l6)
lstm.compile(loss='sparse_categorical_crossentropy', optimizer=tensorflow.keras.optimizers.Adam(),metrics=['accuracy'])
lstm.summary()


# In[14]:


data = pd.read_csv('seq_train.csv', header=None,chunksize = 100000,skiprows=1)


# In[15]:


for df in data:
    df=df[1].str.split(' ',expand=True)
    y= df.iloc[: , 256:]
    x= df.iloc[: , :256]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3, random_state=4)

    x_train = np.asarray(x_train,dtype='float32')
    x_test = np.asarray(x_test,dtype='float32')
    y_train = np.asarray(y_train,dtype='float32')
    y_test = np.asarray(y_test,dtype='float32')
    Y = [int(i) for i in y_train]
    Y = np.asarray(Y,dtype='float32')


    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3,min_delta=1e-4, mode='min', verbose=1)
    stop_alg = EarlyStopping(monitor='val_loss', patience=7,restore_best_weights=True, verbose=1)


    hist = lstm.fit(x_train, Y , batch_size=64, epochs=3, shuffle=True,validation_data = (x_test,y_test))


# In[16]:


fig = plt.figure(figsize=(10,6))
plt.plot(hist.history['loss'], color='#785ef0')
plt.plot(hist.history['val_loss'], color='#dc267f')
plt.title('Model Loss Progress')
plt.ylabel('Categorical Cross-Entropy Loss')
plt.xlabel('Epoch')
plt.legend(['Training Set', 'Test Set'], loc='upper right')
plt.show()


# In[17]:


lstm.save("version_2.h5")


# In[18]:


def prediction(x_test):
    y_pred = model.predict(x_test)
    pred = list(np.argmax(y_pred, axis=1))
    return pred


# In[19]:


test_data = pd.read_csv("seq_test.csv", header=None,chunksize = 100000,skiprows=1)
count=0
for df in test_data:
    result=[]
    df=df[1].str.split(' ',expand=True)
    x = df.to_numpy()
    x = x.astype('float32')
    filename= "test"+str(count)+".csv"
    result=prediction(x)
    df_result = pd.DataFrame(result)
    df_result.to_csv(filename)
    count=count+1


# In[20]:


import glob


# In[21]:


all_files = glob.glob("C:\Users\sadia_tisha1\Desktop\NLP Assignment" + "/*.csv")


# In[22]:


df_from_each_file = (pd.read_csv(f, sep=',') for f in all_files)
df_merged   = pd.concat(df_from_each_file, ignore_index=True)
df_merged.to_csv( "samplesubmission.csv")

