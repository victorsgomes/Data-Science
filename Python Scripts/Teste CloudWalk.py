#%%
print('Teste CloudWalk...')
#%%
result=(10000/(10000+8000+7000))
print(result)
#%%

Holanda=3.5
America_latina=2
Tot=8
print(Holanda+America_latina-Tot)

#%%

def tuple_slice(startIndex, endIndex, tup):
    return None

if __name__ == "__main__":
    print(tuple_slice(1, 4, (76, 34, 13, 64, 12)))
    

#%%
x=(76, 34, 13, 64, 12)    
#%%
y=''

#%%
aux=list(range(2,5,1))

for i in aux:
    if i==0:
        y=y+str(x[1])
    elif i=aux[-1]:
        y=y+x[i]
        

    