
from SuffixAutomaton import SuffixAutomaton


def sam_similar(sam,a,b):
    spans=sam.lcs1(b)
    if not spans:
        return 0 
    (t, start, cand_start)=spans[0]
    if len(t.encode('utf-8'))>=9:
        return 1
    return 0

def doc2pairs(doc,repeat=1):
    used=[0]*len(doc)
    paris=[]
    for i in range(len(doc)-1):
        p=doc[i]
        sam=SuffixAutomaton(p)
        for j in range(i+1,len(doc)):
            if used[i]>=repeat or used[j]>=repeat:
                break
            q=doc[j]
            if sam_similar(sam,p,q):
                paris.append((i,j))
                used[i]+=1
                used[j]+=1
                break
    return paris

