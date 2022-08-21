import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn import set_config, svm
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, KFold
import os
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from scipy.stats import ks_2samp

def transform_time(X):
    """Convert Datetime objects to seconds for numerical/quantitative parsing"""
    df = pd.DataFrame(X)
    return df.apply(lambda x: pd.to_datetime(x).apply(lambda x: x.timestamp()))
                                                      
dir_orig=str(os.path.dirname(os.path.realpath(__file__)))
Base_recov = pd.read_csv(dir_orig+'/Amostra formatada para execução do modelo.csv',sep=";",index_col='Chave')
#%%
print(Base_recov.dtypes)
#print(X_train_SMOTE.shape)
#%%Filtrando as variáveis

###Retirando variáveis descritivas e as que não podem ser extraídas na validação##
Base_recov=Base_recov.drop(columns=['IdContatoSIR','CPF','Nome_Cliente / Empresa','Numero_Contrato','VlDividaAtualizado','SubTipo Produto'])
#%%
###Colocando os valores de data no formato correto##
Base_recov['Data de referência']=pd.to_datetime(Base_recov['Data de referência'],format='%Y-%m-%d %H:%M:%S')
Base_recov['Data_Mora']=pd.to_datetime(Base_recov['Data_Mora'],format='%Y-%m-%d %H:%M:%S')

                                                      
Base_recov['Data de referência']=transform_time(Base_recov['Data de referência'])
Base_recov['Data_Mora']=transform_time(Base_recov['Data_Mora'])                                           
#%%
TEST_SIZE = 0.2
RANDOM_STATE = 42
#N_SPLITS = 3

impute = SimpleImputer(strategy='mean')
scaler = MinMaxScaler()
ohe = OneHotEncoder(handle_unknown='ignore')

numeric_feat = ['VlSOP','Aging','Data de referência','Data_Mora']
pipe_numeric_transf = Pipeline([('SimpleImputer', impute),
                               ('MinMaxScaler', scaler)])

categ_feat = ['Class_Carteira','Class_Produto','Class_Portfolio']
pipe_categ_feat = Pipeline([('OneHotEncoder', ohe)])

preprocessor = ColumnTransformer([('Pipe_Numeric', pipe_numeric_transf, numeric_feat),
                                 ('Pipe_Categoric', pipe_categ_feat, categ_feat)],
                                 remainder='passthrough')
#%%Listando os modelos

log_regress_clf = LogisticRegression(random_state=RANDOM_STATE)###Regressão logística###
grad_boost_clf = GradientBoostingClassifier(random_state=RANDOM_STATE)###Gradient Booster##
SVM_clf=svm.SVC(kernel='linear',random_state=RANDOM_STATE,probability=True)###SVM##
RF_clf=RandomForestClassifier(random_state=RANDOM_STATE)###Random Forest##
XGB_clf=xgb.XGBClassifier(use_label_encoder=False,eval_metric='mlogloss',random_state=RANDOM_STATE)###XGBoost###
bagging_clf = BaggingClassifier(random_state = RANDOM_STATE)
lista_modelos=[('LogisticRegression', log_regress_clf),('GradientBoostingClassifier', grad_boost_clf),('Bagging', bagging_clf),('Random_Forest', RF_clf),('XGBoost', XGB_clf)]

#%%
X = Base_recov.drop(columns='Deals_30')
y = Base_recov.loc[:, 'Deals_30']

x_pretransform=preprocessor.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(x_pretransform, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
smt = SMOTE(random_state=RANDOM_STATE)
X_train_SMOTE, y_train_SMOTE = smt.fit_resample(X_train, y_train)

#%%Executando e salvando
dir_retorn=dir_orig+'/Resultado das amostras (com correção)'
#kfold = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

#df_scores = pd.DataFrame()
df_roc=pd.DataFrame()
df_ks=pd.DataFrame()
plt.plot([0,1],[0,1],'k--')
for name,model in lista_modelos:
    print("Executando o modelo "+name+"...")
    #pipe = Pipeline([('Preprocessor', preprocessor), (name, model)])
    #scores = cross_val_score(model, X_train_SMOTE, y_train_SMOTE, cv=kfold)
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
plt.savefig(dir_retorn+'/Curvas_ROC_modelos.png')
plt.show()
#print(f'Modelo com o maior score: {df_scores.mean().idxmax()}\n')
#df_scores.plot.box(figsize=(12, 5), title='Boxplot Scores')
#plt.savefig(dir_base+'/Box_plots_models.png')
#plt.show()    
print("\nSalvando as tabelas...")
df_ks.to_csv(dir_retorn+'/Tabela Recovery (KS).csv',sep=';')
df_roc.to_csv(dir_retorn+'/Tabela Recovery (Métrica de avaliação).csv',sep=';')
#df_scores.to_csv(dir_base+'/Tabela Recovery (Scores Cross Validation).csv',sep=';')

#%%Tentar rodar o SVM e encaixar nas tabelas (caso dê certo)

