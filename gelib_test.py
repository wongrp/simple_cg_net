import cnine
from gelib import * 


F = SO3vecArr.randn(1,[2],[1,1],device = "cuda")
C = torch.ones(2,2,2)
C[1,1,0] = 2
C = cnine.Rmask1(C)
G = F.gather(C)



print("F = {}".format(F))
print("G = {}".format(G)) 

FcgF = CGproduct(F,F)



