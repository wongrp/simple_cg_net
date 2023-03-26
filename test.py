import torch
import cnine 
import gelib    

X=torch.randn([4,2,2,3,1]) # X1, X2, X3, channel 

p = gelib.SO3partArr.spharm(1,X, device = 'cuda:0')

def f(b,i,j):
    return gelib.SO3part.spharm(1,X[b,i,j,0],X[b,i,j,1],X[b,i,j,2])

print(p)
for b in range(4):
    for i in range(2):
        for j in range(2):
            print(f(b,i,j))
