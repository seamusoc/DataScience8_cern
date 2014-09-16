import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.api as sm
from sklearn import ensemble


def hello():
   print "hello:"

def ams(s,b):
    print "s and b",s,"-",b
    from math import sqrt,log
    if b==0:
        return 0

    return sqrt(2*((s+b+10)*log(1+float(s)/(b+10))-s))



def write_final_file():
  df_test='/Users/seamusocuinneagain/projects/datascience_class/cern/examples/cern.csv'
  #print len(df_cern)-1
  df_cern = pd.read_csv(df_test, index_col=0)
  df_cern['Class']=df_cern['Class'].astype('str')
  df_cern['RankOrder'] =range (1,len(df_cern)+1)
  df_cern = df_cern.sort(axis =0)
  df_cern  =  df_cern[['RankOrder','Class']]
  print df_cern.head()
  df_cern.to_csv(df_test)
  print "file reindexed and written to cern.csv,ready to upolad"

def strip_nan_cols( inframe):
    testline =inframe[3:4]
    testline=testline.applymap(lambda x: np.nan if x==-999.0 else x)
    testline=testline.dropna(axis=1,how='all')
    return testline.columns


def load_raw_data():
  print ("load raw data")  
  load_test_data()
  load_train_data () 
  load_pri_data()

def load_test_data():
   print("load test data") 
   global df_test
   testdatafile='/Users/seamusocuinneagain/projects/datascience_class/cern/examples/test.csv'
   df_test = pd.read_csv(testdatafile, index_col=0)
   
   print ("loaded",len (df_test.index))
    
def load_train_data():
   global df_train
   traindatafile='/Users/seamusocuinneagain/projects/datascience_class/cern/data/training.csv'
   df_train = pd.read_csv(traindatafile, index_col=0)
   df_train = df_train.apply(convert_signal_to_boolean,axis=1)  
   print ("loaded",len (df_train.index))
    
def load_pri_data():
   global df_pri_train
   global df_pri_test
   df_pri_test = df_test.ix[:,'PRI_tau_pt':]
   df_pri_train = df_train.ix[:,'PRI_tau_pt':]
        

def convert_signal_to_boolean(sig):
    if sig['Label']=='s':  
        sig['Label']=1
        return sig
    else:
        sig['Label']=0
        return sig
    


def strip_nan( inframe):
   print ("strip nan cols from frame and return")
   xcols=strip_nan_cols(inframe)
   inframe = inframe[xcols]
   return inframe




def train_boost(traintest,desc,jetno):
   global clf 
   a=traintest
   b=desc
   c=jetno
   X,y,z =get_main_df(a,b,c)
     
   print ("set up train arrays ")
   #X = df_train
   
   
   print y.shape
   print X.shape
   X = X.astype(np.float32)
  
   

   offset = int(X.shape[0] * 0.9)
   print offset ,'offset'
   X_train, y_train,z_train = X[:offset], y[:offset],z[:offset]
   X_test, y_test,z_test = X[offset:], y[offset:], z[offset:]

   print ("xtrain shape", X_train.shape , "ytrain shape" , y_train.shape )
   print ("xtest shape", X_test.shape , "ytest shape" , y_test.shape )
    #xxx
   params={'max_depth': 8, 'subsample': 0.5, 'min_samples_leaf': 35, 'learning_rate': 0.01,'verbose':1}
   #params = {'n_estimators': 100, 'max_depth': 9, 'min_samples_split': 1,
    #      'learning_rate': 0.01,'loss': 'ls','max_features' : 1.0,'min_samples_leaf':35,'verbose':1}
   clf = ensemble.GradientBoostingClassifier(**params)

   clf.fit(X_train, y_train)
   print "score"
   print clf.score(X_train, y_train,z_train)
    
   import matplotlib.pyplot as plt
 
# compute test set deviance
   params = {'n_estimators': 10, 'max_depth': 9, 'min_samples_split': 1,
          'learning_rate': 0.01,'loss': 'ls','max_features' : 1.0,'min_samples_leaf':35,'verbose':1}

  
 
   fig = plt.figure(figsize=(8, 5))
     
    # plot training and testing data
   plt.scatter(X_train, y_train, s=10, alpha=0.4)
   plt.scatter(X_test, y_test, s=10, alpha=0.4, color='red')
   plt.xlim((0, 10))
   plt.ylabel('y')
   plt.xlabel('x')
    
 


def run_cern_output(desc,jet_num,est):
   
  import pickle 
  
  description = desc
  jetnum = jet_num
  estimators=est  
    
  X,y,z =get_main_df('train',description,jetnum)
    
  # print("len x,y,x",X," ",y," ",z)  
    
  pickle_filename =  'pickle_'+description+'.pic'
  train_result_filename = 'cern_'+description +'.out' 
  #params = {'n_estimators': estimators, 'max_depth': 9, 'min_samples_split': 1,
  #        'learning_rate': 0.01,'loss': 'ls','max_features' : 1.0,'min_samples_leaf':35,'verbose':1}

   #Try to improve score
  params = {'n_estimators': estimators, 'max_depth': 15, 'max_features' : 1.0,
          'learning_rate': 0.01,'loss': 'deviance','min_samples_leaf':35,'subsample': 0.3,'verbose':1}  
  params={'n_estimators': estimators,'max_features': 0.1, 'learning_rate': 0.01, 'max_depth': 10, 'min_samples_leaf': 11,'verbose':1}
  params={'n_estimators': 53,'max_features': 0.1, 'learning_rate': 0.01, 'max_depth': 12, 'min_samples_leaf': 11,'verbose':1}
  params={   'n_estimators': 300, 'max_depth': 7, 'min_samples_leaf':200,'loss': 'deviance','verbose':1}
# jet1 best params = {'n_estimators': 150, 'max_depth': 7, 'min_samples_leaf':200,'loss': 'deviance','verbose':1}
#jet2 best 7 300 
  
 # clf = ensemble.GradientBoostingRegressor(**params)
  clf = ensemble.GradientBoostingClassifier(**params)
  print "fit data"
  print X.shape
  print "y shape"  
  print y.shape
  clf.fit(X, y)
  
 # for pred in clf.staged_predict(X):
  #   plt.plot(X[:,0],pred,color='r',alpha=0.1)

        
        
        
  
  filehandler = open(pickle_filename, 'wb')     
  pickle_file=pickle.dump(clf,filehandler,-1) #use latest protocol 
  filehandler.close()
    
  filehandler = open(pickle_filename, 'rb') 
  clf2=pickle.load(filehandler)
    
# score
#get train data  note y is empty labels only supplied for training file
  X,y,z =get_main_df('test',description,jetnum)
  x_result= clf2.predict(X)
  print ("xresult",x_result)
  print (x_result[3],x_result[4],x_result[5])
  print "len of result ",len(x_result)
  print "result shape:",x_result.shape  

    
  # create output format
    
  dfid = pd.DataFrame()
  dfid['id']=X.index.values

  dfcern = pd.DataFrame()
  dfcern['result']=x_result
  dfcern['id']=dfid['id']

#EventId,RankOrder,Class

  dfcern['Label']=''
  dfcern['Label'] = dfcern['result'].apply(lambda x: 's' if x >=0.80  else 'b')

  print (dfcern.head()) 
#send results to file
  dfkaggle = dfcern[['id','Label']]
  dfkaggle.set_index(['id'], inplace = True)                    
  dfkaggle.head(20)
  outfile = '/Users/seamusocuinneagain/projects/datascience_class/cern/examples/'+desc+'.csv'
  dfkaggle.to_csv(outfile)

    
  print ("done")




def run_all(treedepth):
   tree_depth=treedepth
   print ("Run for all #jet and nomass with tree_depth:-",tree_depth) 
   run_cern_output('0jet',0,tree_depth)
   run_cern_output('1jet',1,tree_depth)
   run_cern_output('2jet',2,tree_depth)
   run_cern_output('3jet',3,tree_depth)
   run_cern_output('0mass',0,tree_depth)  




def get_main_df(test_or_train,desc,jet_num):
   # totals for testfile good print (162712+153003+104469+45994+83822)
    global train_labels
    global train_weights
    
    print ("get data frame for " , test_or_train,desc,jet_num)  

    if(test_or_train=="train"):
       if (desc=="0mass"): 
          dfmain=df_train [ df_train['DER_mass_MMC']==-999.0 ]
          del dfmain['DER_mass_MMC']
       else:
          dfmain=df_train[df_train['PRI_jet_num']==jet_num]       
          dfmain=dfmain[dfmain['DER_mass_MMC']>0]  

       print "set train labels"
       train_weights =  dfmain['Weight'].values       
       train_labels = dfmain['Label'].values   
#drop weight and label col!!!                
       dfmain = dfmain.ix[:,:'PRI_jet_all_pt']        
       print ("number of train labels",len(train_labels))
    elif (test_or_train=="test"):
       if (desc=="0mass"):    
          dfmain=df_test[ df_test['DER_mass_MMC']==-999.0 ]
          del dfmain['DER_mass_MMC']
       else:
          dfmain=df_test[df_test['PRI_jet_num']==jet_num]       
          dfmain=dfmain[dfmain['DER_mass_MMC']>0] 

    
       
    print ("Strip nan values")
    cols = strip_nan_cols(dfmain)
    dfmain = dfmain[cols]        

   
    return (dfmain,train_labels,train_weights)
