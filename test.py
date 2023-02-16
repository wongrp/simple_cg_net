import torch
import cnine 
import gelib    

X=torch.randn([2,1,3,1]) # X1, X2, X3, channel 

p=gelib.SO3partArr.spharm(1,X)

f = gelib.SO3part.spharm(1,X.squeeze(1))

print(p)
print(f)