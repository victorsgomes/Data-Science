import os
import pandas as pd

dir_orig=str(os.path.dirname(os.path.realpath(__file__)))
file=dir_orig+'/'+input('Digite o nome do arquivo (incluindo o formato dele): ')
Tab_arq=pd.read_table(file,sep="\n")
nome_col=input("Digite o nome da coluna que contém os CNPJs: ") ###Verificar formato da tabela de retorno para caso seja necessário alguma alteração
Tab_arq=Tab_arq[nome_col].astype(str).values.tolist()

##Criando a chave

import random
import string

def get_random_string(length):
    # With combination of lower and upper case
    result_str = ''.join(random.choice(string.ascii_letters) for i in range(length))
    # print random string
    return(result_str)

# string of length 8
chave_name=get_random_string(15)


from itsdangerous import URLSafeTimedSerializer

s=URLSafeTimedSerializer(chave_name)

aux=s.dumps(Tab_arq)

if not os.path.exists(dir_orig+'/Dados criptografados') :
    os.makedirs(dir_orig+'/Dados criptografados')

from datetime import date
file_cript=open(dir_orig+'/Dados criptografados'+'/Arquivo criptografado ('+date.today().strftime("%d-%m-%Y")+').txt','w')
file_cript.write(aux)
file_cript.close()

file_key=open(dir_orig+'/Chave criptografia.txt','w')
file_key.write(chave_name)
file_key.close()