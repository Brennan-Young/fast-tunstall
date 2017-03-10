import tunstall_code as t
import math
import numpy as np
import matplotlib.pyplot as plt

M,K,EL = t.tunstallNodes(.89,50)
lM = []
mK = []
for m in M:
    lM.append(math.log(m,2))
    
#print lM
    
for (lm,k) in zip(lM,K):
    mK.append(lm/float(k))

mK2 = []
trK = t.achievabilityTrivial(M)
for (lm,k) in zip(lM,trK):
    mK2.append(lm/float(k))

mKg = []
gK = t.gaussApprox(M,.89,.001)
for (lm,k) in zip(lM,gK):
    mKg.append(lm/float(k))
    
cR = []
for (lm,el) in zip(lM,EL):
    cR.append(lm/el)

plt.figure(figsize=(20,10))
plt.axis([40, 160, .5, 1.1])
plt.plot(lM,mK,label = 'Tunstall')
plt.plot(lM,mK2,label = 'Trivial Achievability')
plt.plot(lM,mKg,label = 'Gaussian Approximation')
plt.xlabel(r'$\log_2M$')
plt.ylabel(r'$\frac{\log_2M}{k_{\max}}$')
plt.legend(loc = 4)
plt.show()

plt.figure(figsize=(20,10))
plt.plot(lM,cR)
plt.show()