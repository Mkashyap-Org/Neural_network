import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools
from sklearn.utils import shuffle
import math
import random
from random import seed
from random import randrange

def sigmoid(z):
  return 1/(1 + np.exp(-z))

def dSigmoid(Z):
    s = 1/(1+np.exp(-Z))
    dZ = s * (1-s)
    return dZ

def Relu(Z):
    return np.maximum(0,Z)

def dRelu(x):
    x[x<=0] = 0
    x[x>0] = 1
    return x

def initialize_parameters(n_x, n_h1,n_h2, n_y):
  W1 = np.random.randn(n_h1, n_x)/np.sqrt(2/n_x)
  b1 = np.zeros((n_h1, 1))
  W2 = np.random.randn(n_h2, n_h1)/np.sqrt(2/n_h1)
  b2 = np.zeros((n_h2, 1))
  W3 = np.random.randn(n_y, n_h2)/np.sqrt(2/n_h2)
  b3 = np.zeros((n_y, 1))  

  parameters = {
    "W1": W1,
    "b1" : b1,
    "W2": W2,
    "b2" : b2,
    "W3": W3,
    "b3" : b3 
  }
  return parameters	
  
def forward_prop(X, parameters):
  W1 = parameters["W1"]
  b1 = parameters["b1"]
  W2 = parameters["W2"]
  b2 = parameters["b2"]
  W3=  parameters["W3"]
  b3=  parameters["b3"]
  
  Z1 = np.dot(W1, X) + b1  
  A1 = sigmoid(Z1)
  Z2 = np.dot(W2, A1) + b2
  A2 = sigmoid(Z2)
  Z3 = np.dot(W3, A2) + b3
  A3 = sigmoid(Z3)  
  cache = {
    "A1": A1,
    "A2": A2,
    "A3": A3,
    "Z1": Z1,
    "Z2": Z2,
    "Z3": Z3  
  }
  return A3, cache  

def calculate_cost(A3, Y,m):
  cost = -np.sum(np.multiply(Y, np.log(A3)) +  np.multiply(1-Y, np.log(1-A3)))/m
  cost = np.squeeze(cost)

  return cost

def backward_prop(X, Y, cache, parameters):
  A1 = cache["A1"]
  A2 = cache["A2"]
  A3 = cache["A3"]  

  W2 = parameters["W2"]
  W3=  parameters["W3"]


  Yh=A3
  dLoss_Yh = - (np.divide(Y, Yh ) - np.divide(1 - Y, 1 - Yh))    

  dLoss_Z3 = dLoss_Yh * dSigmoid(cache['Z3'])    
  dLoss_A2 = np.dot(parameters["W3"].T,dLoss_Z3)
  dLoss_W3 = 1./cache['A2'].shape[1] * np.dot(dLoss_Z3,cache['A2'].T)
  dLoss_b3 = 1./cache['A2'].shape[1] * np.dot(dLoss_Z3, np.ones([dLoss_Z3.shape[1],1]))

  dLoss_Z2 = dLoss_Yh * dSigmoid(cache['Z2'])    
  dLoss_A1 = np.dot(parameters["W2"].T,dLoss_Z2)
  dLoss_W2 = 1./cache['A1'].shape[1] * np.dot(dLoss_Z2,cache['A1'].T)
  dLoss_b2 = 1./cache['A1'].shape[1] * np.dot(dLoss_Z2, np.ones([dLoss_Z2.shape[1],1])) 

  dLoss_Z1 = dLoss_A1 * dSigmoid(cache['Z1'])        
  dLoss_A0 = np.dot(parameters["W1"].T,dLoss_Z1)
  dLoss_W1 = 1./X.shape[1] * np.dot(dLoss_Z1,X.T)
  dLoss_b1 = 1./X.shape[1] * np.dot(dLoss_Z1, np.ones([dLoss_Z1.shape[1],1]))


  grads = {
    "dW1": dLoss_W1,
    "db1": dLoss_b1,
    "dW2": dLoss_W2,
    "db2": dLoss_b2,
    "dW3": dLoss_W3,
    "db3": dLoss_b3  
  }

  return grads

def update_parameters(parameters, grads, learning_rate):
  W1 = parameters["W1"]
  b1 = parameters["b1"]
  W2 = parameters["W2"]
  b2 = parameters["b2"]
  W3 = parameters["W3"]
  b3 = parameters["b3"]
    
  dW1 = grads["dW1"]
  db1 = grads["db1"]
  dW2 = grads["dW2"]
  db2 = grads["db2"]
  dW3 = grads["dW3"]
  db3 = grads["db3"]
    
  W1 = W1 - learning_rate*dW1
  b1 = b1 - learning_rate*db1
  W2 = W2 - learning_rate*dW2
  b2 = b2 - learning_rate*db2
  W3 = W3 - learning_rate*dW3
  b3 = b3 - learning_rate*db3  
  
  new_parameters = {
    "W1": W1,
    "W2": W2,
    "W3": W3,  
    "b1" : b1,
    "b2" : b2,
    "b3" : b3  
      
  }

  return new_parameters

def getInAndOutData(data):
    X=data.iloc[:,0:2].values.transpose()
    Y=data.iloc[:,2:].values.transpose()

    return X,Y
    
def divideDataIntoBatches(scaled_df):   
        x= scaled_df
        k=int(math.ceil(len(x)/50))
        foldsize=50      
        folds=[x[(i*foldsize):(i*foldsize)+foldsize] for i in range(0,k)]
        
        return folds
def transformData(df):                 
    df.iloc[:,3].replace(-1, 0,inplace=True)

    scaler = MinMaxScaler() 
    scaled_df = scaler.fit_transform(df.iloc[:,0:4]) 
    scaled_df = pd.DataFrame(scaled_df) 
    
    return scaled_df

def splitData(scaled_df):
    data_length=len(scaled_df[0])                 
    train_datlen = int(data_length*.95)                 

    x=scaled_df.iloc[0:train_datlen,1:4]
    xval=scaled_df.iloc[train_datlen:data_length,1:4]
    return x,xval    

def checkIfGreater(lis,num):
    high = False
    lis1=[]
    if(len(lis)>5):
        for i in range(0,5):
            lis1.append(lis.pop())
        bool_lis = [num >= i for i in lis1]    
        if (all(bool_lis)):
            high = True
            
    return high

def pred(X, Y, parameters):  
        
        comp = np.zeros((1,X.shape[1]))
        a3, cache = forward_prop(X, parameters)   
        yhat=a3
        for i in range(0, yhat.shape[1]):
            if yhat[0,i] > 0.5: comp[0,i] = 1
            else: comp[0,i] = 0
        accuracy=np.sum((comp == Y)/X.shape[1])
        print("Acc: " + str(accuracy))
        
        return comp,accuracy

def sample_floats(low, high, k=1):
    result = []
    seen = set()
    for i in range(k):
        x = random.uniform(low, high)
        while x in seen:
            x = random.uniform(low, high)
        seen.add(x)
        result.append(x)
    return result		

def model(df, n_x, n_h1,n_h2, n_y, num_of_iters, learning_rate):
    train_Acc=[]
    val_Acc=[]
    errors_list=[]
    itr=[]
    final_val_error=0
    parameters = initialize_parameters(n_x, n_h1,n_h2, n_y)
    scaled_df = transformData(df)
    cost=0
    x,xval= splitData(scaled_df)
    for i in range(0, num_of_iters+1):
        folds = divideDataIntoBatches(x) 
        for data in folds:
            m = data.shape[1]
            X,Y=getInAndOutData(data)
              
            a3, cache = forward_prop(X, parameters)

            cost = calculate_cost(a3, Y,m)

            grads = backward_prop(X, Y, cache, parameters)

            parameters = update_parameters(parameters, grads, learning_rate)

        if(i%250 == 0):
            print('Cost after iteration# {:d}: {:f}'.format(i, cost))  
            errors_list.append(cost)
            final_val_error=cost
            itr.append(i)
            x_train,y_train= getInAndOutData(x)
            comp_train,acc_train = pred(x_train,y_train,parameters)
            train_Acc.append(acc_train)
            x_validation,y_validation= getInAndOutData(xval)
            comp_val,acc_val = pred(x_validation,y_validation,parameters)
            val_Acc.append(acc_val)
            if(checkIfGreater(errors_list,cost)):
                print('breaking')
                break            
    return parameters,itr,val_Acc,train_Acc,final_val_error

def train():
	np.random.seed(2)
	# Set the hyperparameters
	n_x = 2     
	n_h1 = 5
	n_h2 = 5   
	n_y = 1     
	num_of_iters = 100000
	learning_rate = 5/num_of_iters	
	df = pd.read_csv('DWH_Training.csv',header=None)
	trained_parameters,itr,val_Acc,train_Acc,last_error = model(df, n_x, n_h1,n_h2, n_y, num_of_iters, learning_rate)
	plt.plot(itr, train_Acc,'-b',label='Training Accuracy')
	plt.plot(itr, val_Acc,'-r', label='Validation accuracy')
	plt.title('Accuracy plot of training and validation Data')
	plt.xlabel('iterations')
	plt.ylabel('Accuracy')
	plt.legend()
	plt.show()	
	
def test():
	np.random.seed(2)
	# Set the hyperparameters
	n_x = 2     
	n_h1 = 5
	n_h2 = 5   
	n_y = 1     
	num_of_iters = 100000
	learning_rate = 5/num_of_iters	
	df = pd.read_csv('DWH_Training.csv',header=None)
	lambda_list = sample_floats(10./num_of_iters, high=20.0/num_of_iters, k=11)
	error_dictionary={}
	for lam in lambda_list:
		trained_parameters,itr,val_Acc,train_Acc,final_val_error = model(df, n_x, n_h1,n_h2, n_y, num_of_iters, lam)
		error_dictionary[final_val_error] =[lam,trained_parameters]
	yy = list(error_dictionary)
	s_list=sorted(yy)
	lr_parameter=error_dictionary[s_list[0]][1]
	df_test = pd.read_csv('DWH_test.csv',header=None)
	test_sc_df = transformData(df_test)
	x_t,y_t=getInAndOutData(test_sc_df)
	pred(x_t,y_t,lr_parameter)
	xx=[error_dictionary[d][0] for d in yy]
	plt.plot(xx,yy,'bs')
	plt.xticks(rotation=90)
	plt.title('learning Rate Vs validation error')
	plt.xlabel('learning Rate')
	plt.ylabel('validation Error')
	plt.show()	
	
	
	
	
train()
test()	