import pymongo
import pandas as pd
import os
import sys
import getpass
from datetime import datetime
from datetime import timedelta
from itertools import chain

##Selecionando o diretório original##
dir_orig=str(os.path.dirname(os.path.realpath(__file__)))

###Pedindo as entradas ao usuário
usuario=input("\nDigite o usuário: ")
senha=getpass.getpass("\nDigite a senha: ")
time_out=int(input("\nDigite o time out (em minutos): " ))
time_out=str(time_out*60000) 

print("\nCarregando as bases...")
client=pymongo.MongoClient("mongodb://"+usuario+":"+senha+"@qq-prd-shard-00-00-qxofh.azure.mongodb.net:27017,qq-prd-shard-00-01-qxofh.azure.mongodb.net:27017,qq-prd-shard-00-02-qxofh.azure.mongodb.net:27017,qq-prd-shard-00-03-qxofh.azure.mongodb.net:27017/test?ssl=true&replicaSet=qq-prd-shard-0&authSource=admin&retryWrites=true&readPreference=secondary&readPreferenceTags=nodeType:ANALYTICS&w=majority&sockettimeoutms="+time_out)
db=client["qq"]

print("Selecionando os credores...")
col_creditor=db["col_creditor"]

Query_credores=[
	{
		"$match" : {
			"active" : True,
			"_id" : {"$nin" : ["pernambucanas","fake"]}
		}
	},
	{
		"$project" : {
			"_id" : 0,
			"Credores" : "$_id"
		}
	}
]

credores=pd.DataFrame(list(col_creditor.aggregate(pipeline=Query_credores,allowDiskUse=True)))



while True:  
    print("\n TABELA DOS CREDORES: \n",credores)
    select_credores=input("Digite o indíce dos credores desejados separados por vírgula: ").split(",")
    select_credores=list(map(str.strip,select_credores))
    select_credores=list(map(int,select_credores))
    credores_select=credores.loc[credores.index[select_credores]] 
    
    print("\nOs credores selecionados foram: \n",credores_select)
    resp=input("\n\nEstá correto (S/N)? ")
    while resp.upper() not in ['S','N']:
        print('\nResposta inválida, responda com "S"ou "N".')
        resp=input("\n\nEstá correto (S/N)? ")
    if (resp.upper()=='S'):
        break
    else:
        print("\nTentando novamente...")

del credores,resp,col_creditor

credores_select=credores_select['Credores'].tolist()   

col_person=db["col_person"]

print("\nSelecionando as datas limites...")
##Datas como input##
data_inic=datetime.strptime(input('Digite a data inicial (formato dd/mm/YYYY): ')+' 03:00:00',"%d/%m/%Y %H:%M:%S") ##Aqui estão as datas limites
data_fim=datetime.strptime(input('Digite a data final (formato dd/mm/YYYY): ')+' 03:00:00',"%d/%m/%Y %H:%M:%S")+ timedelta(days=1)

print('\nConstruindo a query para seleção dos CPFs que cravaram ou quitaram acordo dos credores selecionados no período ...')

Query_deals_infos=[
	{
		"$match" : {
				"$or" : [
						{
							"deals" : {
								"$elemMatch" : {
									"createdAt" : {"$gte" : data_inic, "$lt" : data_fim},
									"status" : {"$in" : ["settled","promise","active"]},
									"creditor" : {"$in" : credores_select}
								}
							},
							"documentType" : "cpf"
						},
						{
							"deals" : {
								"$elemMatch" : {
									"settledAt" : {"$gte" : data_inic, "$lt" :data_fim},
									"creditor" : {"$in" : credores_select}
								}
							},
							"documentType" : "cpf"
						}
				]	
		}
	},
	{
		"$unwind" : "$deals"
	},
	{
		"$match" : {
				"$or" : [
						{
							"deals.createdAt" : {"$gte" : data_inic, "$lt" : data_fim},
							"deals.status" : {"$in" : ["settled","promise","active"]},
							"deals.creditor" : {"$in" : credores_select}
						},
						{
							"deals.settledAt" : {"$gte" : data_inic, "$lt" : data_fim},
							"deals.creditor" : {"$in" : credores_select}
						}
				]				
			}
	},
	{
		"$addFields" : {
				"pos_name" : { "$ifNull" : [{
												"$map" : {
															"input" : "$info.names.name",
															"as" : "nomes",
															"in" : {"$strLenCP" : "$$nomes"}
															}
														},"NA"]
							},
            "contact_deal" : {"$ifNull" : ["$deals.offer.tokenData.contact",{"$ifNull" : ["$deals.offer.tracking.contact","NA"]}]},
            "other_phone" :  {"$ifNull" :["$deals.contacts.phone","NA"]}
		}
	},
	{
		"$project": {
			"_id" : 0,
			"CPF" : "$document",
			"Name" : {"$cond" : [{"$eq" : ["$pos_name","NA"]},
									"NA",
									{"$arrayElemAt" : ["$info.names.name",{"$indexOfArray" :["$pos_name",{"$max" :  "$pos_name"}]}]}]},
			"contact_deal": "$contact_deal",
			"other_phone" : "$other_phone"
		}
	}
    ]

print("\nExecutando a query...")                                                                                        
Tab_ver=pd.DataFrame(list(col_person.aggregate(pipeline=Query_deals_infos,allowDiskUse=True)))
Tab_info_deals=Tab_ver

print("\nAjustando as informações do contato do acordo...")

Tab_info_deals.loc[Tab_info_deals['contact_deal'].str.contains('@',case=True),'contact_deal']="NA"
Tab_info_deals.loc[Tab_info_deals['contact_deal'].str.contains("^[A-Za-z]+$",case=True),'contact_deal']="NA"
Tab_info_deals.loc[Tab_info_deals['contact_deal']=="",'contact_deal']="NA"
Tab_info_deals.loc[Tab_info_deals['contact_deal'].str.len().isin([13,12]),'contact_deal']=Tab_info_deals.loc[Tab_info_deals['contact_deal'].str.len().isin([13,12]),'contact_deal'].str.slice(start=2)
Tab_info_deals.loc[Tab_info_deals['contact_deal'].str.len()!=11,'contact_deal']="NA"

print("\nAjustando as informações do outro contato de telefone...")

Tab_info_deals.loc[Tab_info_deals['other_phone'].str.contains('@',case=True),'other_phone']="NA"
Tab_info_deals.loc[Tab_info_deals['other_phone'].str.contains("^[A-Za-z]+$",case=True),'other_phone']="NA"
Tab_info_deals.loc[Tab_info_deals['other_phone']=="",'other_phone']="NA"
Tab_info_deals.loc[Tab_info_deals['other_phone'].str.len().isin([13,12]),'other_phone']=Tab_info_deals.loc[Tab_info_deals['other_phone'].str.len().isin([13,12]),'other_phone'].str.slice(start=2)
Tab_info_deals.loc[Tab_info_deals['other_phone'].str.len()!=11,'other_phone']="NA"

print("\nAjustando para uma coluna só...")
Tab_info_deals['Telefone']=Tab_info_deals['contact_deal']+":"+Tab_info_deals['other_phone']
Tab_info_deals=Tab_info_deals.iloc[:,[0,1,4]]

Tab_info_deals.loc[:,'Telefone']=Tab_info_deals.loc[:,'Telefone'].str.split(':')###mensagem de aviso, funciona mas não sei o q é!!
Tab_info_deals=Tab_info_deals.explode('Telefone')

print("\nRetirando os Telefones e nomes inválidos...")
Tab_info_deals=Tab_info_deals.loc[Tab_info_deals['Telefone']!="NA"]
Tab_info_deals=Tab_info_deals.loc[Tab_info_deals['Name']!="NA"]

### Query das informações para retorno 
print("\nSelecionando os telefones dos CPFs não encontrados no acordo...")

Tab_not_found=pd.DataFrame(Tab_ver.loc[-Tab_ver['CPF'].isin(Tab_info_deals['CPF'])])
Tab_not_found=list(set(Tab_not_found['CPF']))

print("\nExecutando a query para retorno das informações...")
partes=list(range(0,len(Tab_not_found),10000))
if(partes[-1]!=len(Tab_not_found)):
    partes.append(len(Tab_not_found))

Tab_infos=[]

for i in range(0,len(partes)-1):
    print("\nExecutando a parte "+str(i+1)+" de "+str(len(partes)-1)+"...")    
    docs=Tab_not_found[partes[i]:(partes[i+1])]
    Query_info_ret=[
    	{
    		"$match" : {
    			"document" : {"$in" : docs}
    		}
    	},
    	{
    		"$addFields" : {
    				"pos_name" : { "$ifNull" : [{
    												"$map" : {
    															"input" : "$info.names.name",
    															"as" : "nomes",
    															"in" : {"$strLenCP" : "$$nomes"}
    															}
    														},"NA"]
    							}
    		}
    	},
    	{
    		"$unwind" : "$info.phones"
    	},
    	{
    		"$match" : {
    			"info.phones.type" : "mobile"
    		}
    	},
    	{
    		"$project": {
    			"_id" : 0,
    			"CPF" : "$document",
    			"Name" : {"$cond" : [{"$eq" : ["$pos_name","NA"]},
    									"NA",
    									{"$arrayElemAt" : ["$info.names.name",{"$indexOfArray" :["$pos_name",{"$max" :  "$pos_name"}]}]}]},
    			"Telefone": {"$concat" : ["$info.phones.areaCode","$info.phones.number"]},
    			"tags" : {
    							"$reduce" : {
    								"input" : {"$ifNull" : ["$info.phones.tags",["NA"]]},
    								"initialValue" : "",
    								"in" : {"$concat" : ["$$value",{ "$cond" : [ { "$eq" : ["$$value", "" ] }, "", "|" ] },"$$this"]}
    							}
    						}
    		}
    	}
    ]
    Tab_infos.append(list(col_person.aggregate(pipeline=Query_info_ret,allowDiskUse=True)))
Tab_infos=pd.DataFrame(list(chain.from_iterable(Tab_infos)))     


print("\nTabela completa, selecionando os melhores tels de cada CPF...")

skips=["skip:hot","skip:alto","skip:medio"]
Tab_env=Tab_infos.loc[Tab_infos['tags'].str.contains(skips[0],case=True)]

Tab_infos=Tab_infos[-Tab_infos['CPF'].isin(Tab_env['CPF'])]

for i in range(1,len(skips)) :
    if(Tab_infos.shape[0]==0) :
        break
    aux=Tab_infos.loc[Tab_infos['tags'].str.contains(skips[i],case=True)]
    if(aux.shape[0]>0):
        Tab_env=Tab_env.append(aux)
    Tab_infos=Tab_infos[-Tab_infos['CPF'].isin(aux['CPF'])]

Tab_env=Tab_env.loc[Tab_env['Name']!="NA"]
Tab_env=Tab_env.drop('tags',axis=1) 
Tab_env.columns=list(Tab_info_deals.columns)
Tab_env=Tab_env.append(Tab_info_deals)
Tab_env=Tab_env.iloc[:,[2,1,0]]
Tab_env.columns=["Celular","Nome completo","CPF"]

print("\nRetirando possíveis repetições...")
Tab_env=Tab_env.drop_duplicates(subset=['Celular','CPF'])

### Conferindo as amostras anteriores
print("\nRetirando os CPFs das amostras anteriores...")

arq=os.listdir(dir_orig+"/Campanhas anteriores")

while True:
    conf_anter=input('As bases selecionandas nas campanhas anteriores serão removidas da base selecionada? \n\n*Não= digite 0;\n*Sim= digite 1;\n\nResposta: ')    
    if conf_anter in ['0','1']:
        conf_anter=bool(int(conf_anter))
        break
    else:
        print("Escolha uma opção válida (0 ou 1)")
        
if conf_anter:
    Base_anter=[]
    if len(arq)>0:
        for i in range(0,len(arq)):
            print("Pegando os CPFs da amostra "+str(i+1)+" de "+str(len(arq))+"...")
            aux=pd.read_table(dir_orig+"/Campanhas anteriores/"+arq[i],sep=";",dtype='str')
            Base_anter.append(aux)
            
    Base_anter=list(set(pd.concat(Base_anter)['CPF']))
    print("\nRetirando da amostra selecionada os CPFs já acionados...")
    Tab_env=Tab_env[-Tab_env['CPF'].isin(Base_anter)]

print("\nSalvando...")

Tab_env.to_csv(dir_orig+"/Informações dos CPFs que pagaram e possuem acordo ativo ("+datetime.today().strftime("%d-%m-%Y")+")[CONTROLE].txt",index=False,sep=";")

Tab_env.to_csv(dir_orig+"/Informações dos CPFs que pagaram e possuem acordo ativo ("+datetime.today().strftime("%d-%m-%Y")+")[ENVIAR].txt",index=False,sep=";")
