#%%Carregando a base
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix
from numpy import percentile
import seaborn as sns
from imblearn.over_sampling import SMOTENC
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, BaggingClassifier
import xgboost as xgb
from sklearn.naive_bayes import GaussianNB
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score,roc_auc_score,roc_curve,precision_recall_fscore_support,silhouette_score
#%%
dir_orig=str(os.path.dirname(os.path.realpath(__file__)))

Data_base=pd.read_csv(dir_orig+'/DataBase.txt',sep=',')
Data_base['Dt_Customer']=pd.to_datetime(Data_base['Dt_Customer'],format='%Y-%m-%d')
#%%Verificando as variáveis

pd.DataFrame(Data_base.dtypes).to_csv(dir_orig+'/tipos das variáveis.csv')
print(Data_base.dtypes)
#%%Verificando os missings

num_missings=[]
for i in Data_base:
    print("Coluna: ",i,", número de missings: ",Data_base[i].isna().sum())
    num_missings.append(Data_base[i].isna().sum())

Tab_missings=pd.DataFrame({'Variables' : list(Data_base.columns),'Num_missings' : num_missings})
Tab_missings.to_csv(dir_orig+'/missings_table.csv',sep=';',index=False)
###Só há missings na variável 'Income'##
#%%Analisando a coluna missing

#Data_base.boxplot(column=['Income'])
q25, q75 = percentile(Data_base['Income'].dropna(), 25), percentile(Data_base['Income'].dropna(), 75)
cut_off = (q75 - q25) *1.5
upper=q75+cut_off
Data_base[Data_base['Income']< cut_off]['Income'].hist(figsize=(1,1),bins=100)##Base assimétrica, portanto vamos preencher os outliers com a mediana

#%%Criando a variável 'aging'
Data_base['aging']=2022-Data_base['Year_Birth']

#%%Verificando as proporções das variáveis respostas
resposta_data_base=pd.DataFrame(data={'Contagem' : Data_base['Response'].value_counts(),'Proporção(%)' : Data_base['Response'].value_counts()/Data_base.shape[0]})
print(resposta_data_base)##Base desbalanceada

resposta_data_base.to_csv(dir_orig+'/Proporção_resposta.csv',sep=';')
#%%Dummiezando as variáveis categóricas
categ_variables=['Education','Marital_Status']

Data_categ=pd.get_dummies(Data_base[categ_variables],drop_first=False)
print(Data_categ.columns)
#%%Verificando se há alguma variável com desvio padrão zerado

#print(Data_base.std())

pd.DataFrame(Data_base.std(),columns=['desvio_padrão']).to_csv(dir_orig+'/table_std.csv',sep=';')

###Variáveis Z tem desvio zero, portanto devem ser excluídas
#%%Base final para análise

Data_resposta=Data_base['Response'].astype(bool)
Data_base=Data_base.drop(columns=['Year_Birth','Education','Marital_Status','Dt_Customer','Z_CostContact','Z_Revenue','Response'])
numeric_features=list(Data_base.columns)
Data_base=pd.concat([Data_base,Data_categ],axis=1)

#%%Analisando as correlações
plt.figure(figsize=(8,8))
cmap = sns.diverging_palette(220, 20, sep=20, as_cmap=True)

sns.heatmap(Data_base.corr(),center=0, cmap=cmap,annot=False)

#%%Padronizando as variáveis numéricas

impute = SimpleImputer(strategy='median')
scaler = StandardScaler()
RANDOM_STATE=22
pipe_numeric_transf = Pipeline([('SimpleImputer', impute),
                               ('MinMaxScaler', scaler)])
preprocessor = ColumnTransformer([('Pipe_Numeric', pipe_numeric_transf, numeric_features)],
                                 remainder='passthrough')
Data_base_normalized=Data_base
numeric_tranf=pd.DataFrame(preprocessor.fit_transform(Data_base_normalized[numeric_features]),columns=numeric_features,index=Data_base.index)
for col in numeric_features:
   Data_base_normalized[col]= numeric_tranf[col]

#%%Separando a base de treino e teste
X_train, X_test, y_train, y_test = train_test_split(Data_base_normalized, Data_resposta, test_size=0.2, random_state=22)

#%%Selecionando a técnica de desbalanceamento

index_col_categ=list(range(23,36))
sm_resample = SMOTENC(categorical_features=index_col_categ, random_state=RANDOM_STATE,sampling_strategy = 0.5)

X_train_sm, Y_train_sm=sm_resample.fit_resample(X_train, y_train)

#%%Verificando as variáveis significativas na regressão logística

logit = sm.Logit(Y_train_sm,X_train_sm)
result = logit.fit(maxiter=100,solver='lbfgs')
plt.rc('figure', figsize=(14, 11))
#plt.text(0.01, 0.05, str(result.summary()), {'fontsize': 12}) #old approach
plt.text(0.01, 0.05, str(result.summary()), {'fontsize': 10}, fontproperties = 'monospace') # approach improved by OP -> monospace!
plt.axis('off')
plt.tight_layout()
plt.savefig(dir_orig+'/output_logistic_regression.png')
#%%Selecionando as variáveis significativas do modelo
variables_sign=['Teenhome','Recency','MntMeatProducts','NumDealsPurchases','NumWebPurchases','NumStorePurchases','NumWebVisitsMonth','AcceptedCmp3','AcceptedCmp4','AcceptedCmp5','AcceptedCmp1','Complain','Education_2n Cycle','Education_Basic','Education_PhD','Marital_Status_Divorced','Marital_Status_Married','Marital_Status_Single','Marital_Status_Together','Marital_Status_Widow']

X_train_sm=X_train_sm[variables_sign]
X_test=X_test[variables_sign]
#%%Listando as técnicas a serem testadas

log_regress_clf = LogisticRegression(random_state=RANDOM_STATE)###Regressão logística###
grad_boost_clf = GradientBoostingClassifier(random_state=RANDOM_STATE)###Gradient Booster##
nb_clf=GaussianNB()
RF_clf=RandomForestClassifier(random_state=RANDOM_STATE)###Random Forest##
XGB_clf=xgb.XGBClassifier(use_label_encoder=False,eval_metric='mlogloss',random_state=RANDOM_STATE)###XGBoost###
bagging_clf = BaggingClassifier(random_state = RANDOM_STATE)
lista_modelos=[('LogisticRegression', log_regress_clf),('GradientBoostingClassifier', grad_boost_clf),('Bagging', bagging_clf),('Random_Forest', RF_clf),('XGBoost', XGB_clf),('Naive_Bayes',nb_clf)]

#%%Executando e salvando

df_anals=pd.DataFrame()
#df_ks=pd.DataFrame()
plt.plot([0,1],[0,1],'k--')
for name,model in lista_modelos:
    print("Executando o modelo "+name+"...")
    #pipe = Pipeline([('Preprocessor', preprocessor), (name, model)])
    #scores = cross_val_score(model, X_train_SMOTE, y_train_SMOTE, cv=kfold)
    #df_scores.loc[:, name] = scores
    model.fit(X_train_sm,Y_train_sm)
    y_pred = model.predict(X_test) 
    roc=roc_auc_score(y_test,y_pred)
    y_pred_prob=model.predict_proba(X_test)[:,1]
    fpr , tpr, thresholds = roc_curve(y_test, y_pred_prob)
    plt.plot(fpr,tpr,label=name)
    ks=ks_2samp(y_pred_prob[y_test==True],y_pred_prob[y_test==False]).statistic
    df_anals.insert(df_anals.shape[1],name,[roc,2*roc-1,accuracy_score(y_test, y_pred),precision_recall_fscore_support(y_test,y_pred)[1][0],precision_recall_fscore_support(y_test,y_pred)[1][1],ks])
    #plot_confusion_matrix(model, X_test, y_test)
    #df_ks.insert(df_ks.shape[1],name,[ks])
    
df_anals.insert(df_anals.shape[1],'metrics',['AUC','GINI','accuracy','precision','specificity','KS'])
df_anals.set_index('metrics',inplace=True,drop=True)    
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
#plt.savefig(dir_base+'/Box_plots_models.png')
#plt.show()    
print("\nSalvando as tabelas...")
#df_ks.to_csv(dir_orig+'/Tabela Case Ifood (KS).csv',sep=';')
df_anals.to_csv(dir_orig+'/Tabela Case Ifood (Métrica de avaliação).csv',sep=';')
#df_scores.to_csv(dir_base+'/Tabela Recovery (Scores Cross Validation).csv',sep=';')

#%%Montar as matrizes de confusão

for name,model in lista_modelos:
    print("Executando o modelo "+name+"...")
    #pipe = Pipeline([('Preprocessor', preprocessor), (name, model)])
    #scores = cross_val_score(model, X_train_SMOTE, y_train_SMOTE, cv=kfold)
    #df_scores.loc[:, name] = scores
    model.fit(X_train_sm,Y_train_sm)
    y_pred = model.predict(X_test) 
    plot_confusion_matrix(model, X_test, y_test,cmap=sns.cm.rocket_r)

#%%Selecionando os melhores clusters

clusters_range=range(2,15)
Data_base_normalized=Data_base_normalized[variables_sign]
results=[]
for c in clusters_range:
    clusterer=KMeans(init='k-means++',n_clusters=c,n_init=100,random_state=RANDOM_STATE)
    cluster_labels=clusterer.fit_predict(Data_base_normalized)
    silhouette_avg=silhouette_score(Data_base_normalized,cluster_labels)
    results.append([c,silhouette_avg])

results=pd.DataFrame(results,columns=['num_cluster','silhouette_score'])
pivot_km=pd.pivot_table(results,index='num_cluster',values='silhouette_score')

plt.figure()
sns.heatmap(pivot_km,annot=True,linewidths=0.5,fmt='.3f',cmap=sns.cm.rocket_r)
plt.tight_layout()
#%%Definido o melhor grupo, hora de classificá-los

kmeans_sel=KMeans(init='k-means++',n_clusters=2,n_init=100,random_state=RANDOM_STATE).fit(Data_base_normalized)

Data_Base_Anals=Data_base
Data_Base_Anals['Response']=Data_resposta
Data_Base_Anals['Cluster']=kmeans_sel.labels_

#%%Comparando os grupos quanto a variável resposta

group_anals=Data_Base_Anals.groupby(['Cluster','Response'],as_index=False).count()