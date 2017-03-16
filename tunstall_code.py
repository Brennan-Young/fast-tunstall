import numpy as np
import matplotlib.pyplot as plt
import math
import pdb
from scipy.special import erfinv
# %matplotlib inline

'''
Main source:


[1]: J. C. Kieffer, "Fast Generation of Tunstall Codes," 2007 IEEE International Symposium on Information 
Theory, Nice, 2007, pp. 76-80.
doi: 10.1109/ISIT.2007.4557079
URL: http://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4557079&isnumber=4557062
'''


def sFunction(tau,s = [0],r = [1]):
    n = len(s)
    ints = range(n+1)
    mlist = [(x+1)*tau for x in ints]
    #print(mlist)
    # sequence m_i: from [1] section III
    m = np.floor(mlist)
    mi = [int(x) for x in m]
    #print(mi)
    # temporary seqeunce t(n): [1] section III
    t = [int(x + np.floor((n+1)*tau) - np.floor(n*tau)) for x in s]
    #print(np.floor((n+1)*tau) - np.floor(n*tau))
    #print(t)
    for i in range(len(t)):
        if not (t[i] in mi):
            tL = t[:i]
            tR = t[i:]
            break
        else:
            tL = t
            tR = []
    #print(tL)
    #print(tR)
    sout = tL + [0] + tR
    a = n + 1
    lside = r[:len(tL)]
    rside = r[len(tL):]
    rout = lside + [a] + rside

    return(sout,rout)

def tunstallNodes2(p, imax, inodes = [(0,0)], imin = 0):
    ''' 
    Not currently used.
    Function to generate the internal nodes of a Tunstall code.  Can be done recursively by specifying a 
    previous set of internal nodes and the level to which that set was generated.  

    '''
    s = [0]
    r = [1]
    q = 1-p
    tau = np.log(q)/np.log(p)
    # trivial case
    if imax == 0:
        return inodes
    seq1 = int(np.floor(tau))
    # append (0,0), (1,0), ..., (m_1,0)
    for i in range(seq1):
        inodes.append((i+1,0))
    for i in range(imax):
        print('i = ' + str(i))
        mi = int(np.floor((i+1)*tau))
        mi2 = int(np.floor((i+2)*tau))
        #[s,r] = sFunction(tau,s,r)
        for a,b in zip(s,r):
            inodes.append((a,b))
        inodes.append((mi+1,0))
        for j in range(int(mi2-mi-1)):
            print('j = '+str(j))
            for a,b in zip(s,r):
                inodes.append((a+j+1,b))
            inodes.append((mi + j + 1 + 1,0))
        [s,r] = sFunction(tau,s,r)


    return inodes

def tunstallOneStep(p, i, inodes, s, r, lext, rext, epsilon, M, kList, ELlist):
    ''' 
    Goes through one step of the generation of a Tunstall code.  This is not the generation of a single leaf; rather, it is the generation of collection of leaves, where the leaves generated follows a single step of Kieffer's algorithm.
    '''
    q = 1-p
    tau = np.log(q)/np.log(p)
    mi = int(np.floor((i+1)*tau))
    mi2 = int(np.floor((i+2)*tau))

    # step (s(n,1),r(n,1)), ..., (s(n,n),r(n,n)), (m_n+1,0)
    for a,b in zip(s,r):
        inodes.append((a,b))
        parentsOfLeaves((a,b),lext,rext)
    inodes.append((mi+1,0))
    parentsOfLeaves((mi+1,0),lext,rext)

    # step (s(n,1)+i,r(n,1)), ..., (s(n,n)+i,r(n,n)), (m_n+1+i,0)
    for j in range(int(mi2-mi-1)):
        for a,b in zip(s,r):
            inodes.append((a+j+1,b))
            parentsOfLeaves((a+j+1,b),lext,rext)
        inodes.append((mi + j + 1 + 1,0))
        parentsOfLeaves((mi + j + 1 + 1,0),lext,rext)
        MKadd(lext,rext,epsilon,p,M,kList)
        ELadd(lext,rext,p,ELlist)

    # generate next s(n) and r(n)
    [s,r] = sFunction(tau,s,r)
    return(inodes,s,r)

def tunstallNodes(p, imax, epsilon = 10**(-3)):
	'''
	Computes a tree for a Tunstall code.  The Tunstall tree is constructed with an i.i.d. binary source with probability p, generated using Kieffer's algorithm.  The number of steps of Kieffer's algorithm run is set to imax.  Epsilon is used to compute a parameter of the code but does not affect the generation of the code itself.

	TODO: Move epsilon away from being an argument to this function.  It makes you think that it will affect how the code is generated (it doesn't)
	'''
    inodes = [(0,0)] # current list of internal nodes
    M = []
    kList = []
    ELlist = []
    lext = [(0,0)] # internal nodes which will be p-extended in leaves
    rext = [(0,0)] # internal nodes which will be q-extended in leaves
    s = [0]
    r = [1]
    q = 1-p
    tau = np.log(q)/np.log(p)
    # trivial case
    if imax == 0:
        return inodes
    seq1 = int(np.floor(tau))
    # append (0,0), (1,0), ..., (m_1,0)
    for i in range(seq1):
        inodes.append((i+1,0))
        parentsOfLeaves((i+1,0),lext,rext)
    for i in range(imax):
        [inodes,s,r] = tunstallOneStep(p,i,inodes,s,r,lext,rext,epsilon,M,kList,ELlist)
        #print inodes
        
        #leaves, counts = makeLeaves(lext,rext)
        #M.append(list_nCr(counts))
        #kList.append(findMaxk(leaves,counts,epsilon,p))
        #pdb.set_trace()
    return (M,kList,ELlist)

def parentsOfLeaves(inode,lext,rext):
    # append the newest node to the list of nodes which need left- and right- extensions
    lext.append(inode)
    rext.append(inode)
    # if this node is above another internal node, remove that node from the list
    try:
        lext.remove((inode[0]-1,inode[1]))
    except ValueError:
        pass
    try:
        rext.remove((inode[0],inode[1]-1))
    except ValueError:
        pass
    return

def makeLeaves(lext,rext):
    ext = lext + rext
    counts = []
    leaves = []
    for l in lext:
        leaves.append((l[0]+1,l[1]))
        counts.append((l[0]+l[1],l[0]))
    for r in rext:
        leaves.append((r[0],r[1]+1))
        counts.append((r[0]+r[1],r[0]))
    return leaves, counts

def nCr(n,r):
    # Standard n choose r function
    # make sure long ints are being used
    f = math.factorial
    return f(n) / (f(r)*f(n-r))

def list_nCr(counts):
    '''
    Computes nCr over a list
    '''
    c = []
    for leaf in counts:
        c.append(nCr(leaf[0],leaf[1]))
    return sum(c)

def findMaxk(leaves,counts,epsilon,p):
    q = 1 - p
    lengthClass = []
    for leaf in leaves:
        lengthClass.append(leaf[0]+leaf[1])
    sortedLengthIndices = sorted(range(len(lengthClass)),key=lambda x:lengthClass[x])
    sortedLengths = sorted(lengthClass)
    sortedTypes = [x for (y,x) in sorted(zip(lengthClass,leaves))]
    sortedCounts = [x for (y,x) in sorted(zip(lengthClass,counts))]
    predictedK = sortedLengths[0]
    s = 0
    if epsilon == 1.:
        return sortedLengths[-1]
    while predictedK <= sortedLengths[-1]:
        #print predictedK
        for l,leaf,c in zip(sortedLengths,sortedTypes,sortedCounts):
            if l == predictedK:
                #print l
                #print leaf
                #print c
                # multiplying a very large number (nCr) with a very small one (p^a).  Trying to avoid 
                # problems which may arise.
                t1 =  math.log(nCr(c[0],c[1])) + leaf[0]*math.log(p) + leaf[1]*math.log(q)
                t2 = math.exp(t1)
                s = s + t2
                #print s
        if s > epsilon:
            #maxk = predictedK - 1
            break
        else: 
            predictedK = predictedK + 1
    maxk = predictedK - 1 
    return maxk

def MKadd(lext,rext,epsilon,p,M,kList):
    leaves, counts = makeLeaves(lext,rext)
    M.append(list_nCr(counts))
    kList.append(findMaxk(leaves,counts,epsilon,p))
    return

def expectedLength(leaves,counts,p):
    q = 1 - p
    El = 0
    for l,c in zip(leaves,counts):
        t1 =  math.log(nCr(c[0],c[1])) + l[0]*math.log(p) + l[1]*math.log(q)
        t2 = math.exp(t1) * (l[0] + l[1])
        El = El + t2
    return El

def ELadd(lext,rext,p,ELlist):
    leaves, counts = makeLeaves(lext,rext)
    ELlist.append(expectedLength(leaves,counts,p))
    return


def achievabilityTrivial(M):
    trK = []
    for m in M:
        trK.append(math.floor(math.log(m,2)))
    return trK

def Qinv(x):
    # Inverse Q function.  Highly useful
    return math.sqrt(2)*erfinv(1-2*(x))

def gaussApprox(M,p,epsilon):
    '''
    Gaussian approximation of the compression rate of a Tunstall code.
    '''
    gK = []
    q = 1-p
    H = p*math.log(1./p,2) + q*math.log(1./q,2)
    H2 = p*math.log(1./p,2)**2 + q*math.log(1./q,2)**2
    sig = H2/H**3-1/H
    mu = 1/H
    for m in M:
        gK.append(Qinv(1-epsilon)*math.sqrt(sig*math.log(m,2)) + mu*math.log(m,2))
    return gK

