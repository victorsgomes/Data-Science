#%%Exemplo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import set_config
from sklearn.metrics import classification_report
import os
import pickle

dir_orig=str(os.path.dirname(os.path.realpath(__file__)))
USE_COLS = ['PassengerId', 'Survived', 'Age', 'Fare', 'Embarked']
TEST_SIZE = 0.2
RANDOM_STATE = 42

df = pd.read_csv(dir_orig+'/train.csv', usecols=USE_COLS, index_col='PassengerId')

X = df.drop(columns='Survived')
y = df.loc[:, 'Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

impute = SimpleImputer(strategy='mean')
scaler = StandardScaler()
ohe = OneHotEncoder(handle_unknown='ignore')
grad_boost_clf = GradientBoostingClassifier(random_state=RANDOM_STATE)

numeric_feat = ['Age', 'Fare']
pipe_numeric_transf = Pipeline([('SimpleImputer', impute),
                               ('StandardScaler', scaler)])

categ_feat = ['Embarked']
pipe_categ_feat = Pipeline([('OneHotEncoder', ohe)])

preprocessor = ColumnTransformer([('Pipe_Numeric', pipe_numeric_transf, numeric_feat),
                                 ('Pipe_Categoric', pipe_categ_feat, categ_feat)])

pipeline = Pipeline([('Preprocessor', preprocessor),
                    ('GradientBoostingClassifier', grad_boost_clf)])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)

print(classification_report(y_test, y_pred))
###Salvando o modelo###
pickle.dump(pipeline,open(dir_orig+'/Modelo_exemplo.sav','wb'))
#%%
modelo_load=pickle.load(open(dir_orig+'/Modelo_exemplo.sav', 'rb'))
y_pred_load = modelo_load.predict(X_test)
#y_score=modelo_load.score(X_test,y_test)
print(classification_report(y_test, y_pred_load))
#print(y_score) ##é a acurácia do modelo
#%%Exercícios

###Selecionando outros modelos##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder,MinMaxScaler
from sklearn import set_config, svm
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, KFold
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier,BaggingClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from scipy.stats import ks_2samp

dir_orig=str(os.path.dirname(os.path.realpath(__file__)))
USE_COLS = ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SPLITS = 10

df = pd.read_csv(dir_orig+'/train.csv', usecols=USE_COLS, index_col='PassengerId')

#%%
impute = SimpleImputer(strategy='mean')
scaler = MinMaxScaler(feature_range=(-1,1))
ohe = OneHotEncoder(handle_unknown='ignore')

numeric_feat = ['Age', 'Fare']
pipe_numeric_transf = Pipeline([('SimpleImputer', impute),
                               ('MinMaxScaler', scaler)])

categ_feat = ['Sex','Embarked']
pipe_categ_feat = Pipeline([('OneHotEncoder', ohe)])

preprocessor = ColumnTransformer([('Pipe_Numeric', pipe_numeric_transf, numeric_feat),
                                 ('Pipe_Categoric', pipe_categ_feat, categ_feat)],
                                 remainder='passthrough')

X = df.drop(columns='Survived')
y = df.loc[:, 'Survived']


x_pretransform=preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(x_pretransform, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
#%%
smt = SMOTE(random_state=RANDOM_STATE)
X_train_SMOTE, y_train_SMOTE = smt.fit_resample(X_train, y_train)

###MODELOS##

log_regress_clf = LogisticRegression(random_state=RANDOM_STATE)###Regressão logística###
grad_boost_clf = GradientBoostingClassifier(random_state=RANDOM_STATE)###Gradient Booster##
SVM_clf=svm.SVC(kernel='linear',random_state=RANDOM_STATE,probability=True)###SVM##
RF_clf=RandomForestClassifier(random_state=RANDOM_STATE)###Random Forest##
XGB_clf=xgb.XGBClassifier(use_label_encoder=False,eval_metric='mlogloss',random_state=RANDOM_STATE)###XGBoost###
bagging_clf = BaggingClassifier(random_state = RANDOM_STATE)


lista_modelos=[('LogisticRegression', log_regress_clf),('GradientBoostingClassifier', grad_boost_clf),('SVM', SVM_clf),('Random_Forest', RF_clf),('XGBoost', XGB_clf),('Bagging',bagging_clf)]

#kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

#df_scores = pd.DataFrame()

df_roc=pd.DataFrame()
df_ks=pd.DataFrame()
plt.plot([0,1],[0,1],'k--')
for name,model in lista_modelos:
    #pipe = Pipeline([('Preprocessor', preprocessor), (name, model)])
    #scores = cross_val_score(pipe, X_train, y_train, cv=kfold)
    #df_scores.loc[:, name] = scores
    model.fit(X_train_SMOTE,y_train_SMOTE)
    y_pred = model.predict(X_test) 
    roc=roc_auc_score(y_test,y_pred)
    df_roc.insert(df_roc.shape[1],name,[roc,2*roc-1,accuracy_score(y_test, y_pred),precision_recall_fscore_support(y_test,y_pred)[1][0],precision_recall_fscore_support(y_test,y_pred)[1][1]])
    y_pred_prob=model.predict_proba(X_test)[:,1]
    fpr , tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr,tpr,label=name)
    ks=ks_2samp(y_pred_prob[y_test==True],y_pred_prob[y_test==False]).statistic
    df_ks.insert(df_ks.shape[1],name,[ks])

df_roc.insert(df_roc.shape[1],'metrics',['AUC','GINI','accuracy','precision','specificity'])
df_roc.set_index('metrics',inplace=True,drop=True)

print('------------------------------------------------------------------------\n')
print('Salvando as curvas ROC dos modelos...\n')
plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title('Curvas ROC dos modelos')
plt.savefig(dir_orig+'/Curvas_ROC_modelos.png')
plt.show()
#print(f'Modelo com o maior score: {df_scores.mean().idxmax()}\n')
#df_scores.plot.box(figsize=(12, 5), title='Boxplot Scores')
#plt.savefig(dir_orig+'/Box_plots_models.png')
#plt.show()    
print("\nSalvando as tabelas...")
df_ks.to_csv(dir_orig+'/Tabela Recovery (KS).csv',sep=';')
df_roc.to_csv(dir_orig+'/Tabela Recovery (AUC e GINI).csv',sep=';')
#df_scores.to_csv(dir_orig+'/Tabela Recovery (Scores Cross Validation).csv',sep=';')

#%%
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,precision_score,precision_recall_fscore_support
aux_3=precision_recall_fscore_support(y_test,y_pred)[1][0]
print(precision_recall_fscore_support(y_test,y_pred)[1][0])
#%%
print("\n-----------------------------------------------------------------------")
print("Montando a curva roc dos Modelos...")
SVM_clf=svm.SVC(kernel='linear',random_state=RANDOM_STATE,probability=True)
lista_modelos=[('LogisticRegression', log_regress_clf),('GradientBoostingClassifier', grad_boost_clf),('SVM', SVM_clf),('Random_Forest', RF_clf),('XGBoost', XGB_clf)]

plt.plot([0,1],[0,1],'k--')
for name,model in lista_modelos:
    pipe = Pipeline([('Preprocessor', preprocessor), (name, model)])
    pipe.fit(X_train,y_train)
    y_pred_prob=pipe.predict_proba(X_test)[:,1]
    fpr , tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr,tpr,label=name)

plt.legend()
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title('Curvas ROC dos modelos')

plt.savefig(dir_orig+'/Curvas_ROC_modelos.png')
plt.show()

#%%
SVM_clf=svm.SVC(kernel='linear',random_state=RANDOM_STATE)
pipe = Pipeline([('Preprocessor', preprocessor), ('SVM', SVM_clf)])
pipe.fit(X_train,y_train)
y_pred_prob=pipe.predict_proba(X_test)
y_pred=pipe.predict(X_test)
#%%Montando os modelos com soluções para balanceamento
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn import set_config, svm
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, KFold
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
#from sklearn.feature_extraction.text import CountVectorizer

dir_orig=str(os.path.dirname(os.path.realpath(__file__)))
USE_COLS = ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked']
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SPLITS = 10

df = pd.read_csv(dir_orig+'/train.csv', usecols=USE_COLS, index_col='PassengerId')

impute = SimpleImputer(strategy='mean')
scaler = StandardScaler()
ohe = OneHotEncoder(handle_unknown='ignore')

numeric_feat = ['Age', 'Fare']
pipe_numeric_transf = Pipeline([('SimpleImputer', impute),
                               ('StandardScaler', scaler)])

categ_feat = ['Sex','Embarked']
pipe_categ_feat = Pipeline([('OneHotEncoder', ohe)])

preprocessor = ColumnTransformer([('Pipe_Numeric', pipe_numeric_transf, numeric_feat),
                                 ('Pipe_Categoric', pipe_categ_feat, categ_feat)],
                                 remainder='passthrough')
X = df.drop(columns='Survived')
y = df.loc[:, 'Survived']

x_pretransform=preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(x_pretransform, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

###Ajustando o balanceamento nos dados##
#vectorizer = CountVectorizer()
#vectorizer.fit(X_train.values.ravel())
#X_train=vectorizer.transform(X_train.values.ravel())
#X_test=vectorizer.transform(X_test.values.ravel())
#X_train=X_train.toarray()
#X_test=X_test.toarray()
smt = SMOTE(random_state=RANDOM_STATE)
X_train_SMOTE, y_train_SMOTE = smt.fit_resample(X_train, y_train)

###MODELOS##
log_regress_clf = LogisticRegression(random_state=RANDOM_STATE)###Regressão logística###
grad_boost_clf = GradientBoostingClassifier(random_state=RANDOM_STATE)###Gradient Booster##
SVM_clf=svm.SVC(kernel='linear',random_state=RANDOM_STATE)###SVM##
RF_clf=RandomForestClassifier(random_state=RANDOM_STATE)###Random Forest##
XGB_clf=xgb.XGBClassifier(objective='binary:logistic',random_state=RANDOM_STATE)###XGBoost###
lista_modelos=[('LogisticRegression', log_regress_clf),('GradientBoostingClassifier', grad_boost_clf),('SVM', SVM_clf),('Random_Forest', RF_clf),('XGBoost', XGB_clf)]

kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

df_scores = pd.DataFrame()

for name,model in lista_modelos:
    #pipe = Pipeline([('Preprocessor', preprocessor), (name, model)])
    scores = cross_val_score(model, X_train_SMOTE, y_train_SMOTE, cv=kfold)
    df_scores.loc[:, name] = scores
print('------------------------------------------------------------------------')
print(f'Modelo com o maior score: {df_scores.mean().idxmax()}')
df_scores.plot.box(figsize=(12, 5), title='Boxplot Scores')
plt.show()
