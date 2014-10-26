# ## Tensor SimRank
import time
import numpy as np
import matplotlib.pyplot as plt

from sp3tensor import sp3tensor

def tensort_simrank(W, c = 0.1, e = 0.00001):
    if W.shape[0] != W.shape[1]:
        raise NotImplementedError, 'Now can promise only for square tensors like NxNxK'
    
    I = W.shape[0]
    
    Sk = np.identity(I)
    res = 100
    
    s = time.clock()
    while res > e:
        core = (W.T.tenten(W.T.tenmat3(Sk).transpose(1, 0, 2))).T
        Sk_ = c*core - c*np.diag(np.diag(core)) + np.identity(I)
        res = np.linalg.norm(Sk - Sk_)
        Sk = Sk_
        print res 
    print 'Done is %f CPU-sec' % (time.clock() - s)
    return Sk


# ## Experiments

text = "this very nice dog looks almost like that very cute puppy that looks almost like pretty dog".split()
D = list(set(text))
# print D
text_id = [D.index(w) for w in text]
# print text_id
I = np.max(text_id) + 1
w = 1
A = sp3tensor.zeros((I, I, w*2))

print len(text)
for i in range(1, len(text_id)-w):
    for k in range(-w, w+1):
        if k != 0:
            A[text_id[i], 
              text_id[i+k], 
              (k+w)/2] += 1

def unzero_vec(x):
    r = x[0]
    r[x==0] = 1
    return r

W = sp3tensor.fromlayers([x.astype(float) / unzero_vec(x.sum(axis = 0)) for x in A.tolayers()])

# print W.T.tenten(W).todense()
S = tensort_simrank(W)

print 'valuable: '

results = []

for i in range(S.shape[0]):
    for j in range(i, S.shape[1]):
        if i != j:
            if S[i, j] > 0.0001:
                results.append( ( D[i], D[j], S[i, j] ) )
                
for w1, w2, val in sorted(results, key = lambda x: x[2], reverse = True):
    print w1, '\t', w2, '\t', val


from collections import Counter

import nltk

def build_adjustancy_matrix(aprox_n = 5000):
    c = Counter(open('/home/ben/tmp/en_small_corpra/eng_news_2005_10K-sentences.txt.out').read().split())
    
    l = zip(*c.most_common(aprox_n))[0]
    D = [w[0] for w in nltk.pos_tag(l) if w[1][0] in ['N', 'V', 'W', 'A', 'J']]

    print 'Exmple of words used:', D[:100]

    I = len(D)
    print 'Number of words left:', I

    w = 1
    A = sp3tensor.zeros((I, I, w*2))
    it_n = 0

    for line in open('/home/ben/tmp/en_small_corpra/eng_news_2005_10K-sentences.txt.out'):
        it_n += 1
        if it_n % 300 == 0:
            print it_n, ' ',

#         if it_n > 5000:
#             break

        text_line_id = [D.index(x) if x in D else -1 for x in line.split()]
        for i in range(w, len(text_line_id)-w):
            if text_line_id[i] != -1:
                for k in range(-w, w+1):
                    if k != 0 and text_line_id[i+k] != -1:
                        A[text_line_id[i], 
                          text_line_id[i+k], 
                          (k+w)/2] += 1
        
    return A, D
                        
A, D = build_adjustancy_matrix()


# In[77]:

def unzero_vec(x):
    r = x[0]
    r[x==0] = 1
    return r

def build_simrank(A):
    W = sp3tensor.fromlayers([x.astype(float) / unzero_vec(x.sum(axis = 0)) for x in A.tolayers()])

    S = tensort_simrank(W)
    return S

def find_closest(S, D):
    print 'valuable: '
    results = []

    for i in range(S.shape[0]):
        for j in range(i, S.shape[1]):
            if i != j:
                if S[i, j] > 0.001:
                    results.append( ( D[i], D[j], S[i, j] ) )

    for w1, w2, val in sorted(results, key = lambda x: x[2], reverse = True)[:50]:
        print w1, '\t', w2, '\t', val
        
S = build_simrank(A)
# find_closest(S, D)


# In[104]:

def find_S_vector_decomposition(S):
    print '0-persent', float(np.sum(S == 0)) / np.prod(S.shape)

    n = S.shape[0]
    r = int(0.1*n)

#     Z = np.random.normal(size = n*p).reshape(n, p)
#     B = np.dot(S, Z)

    U, d, V = np.linalg.svd(S, full_matrices=False)
    plt.plot(d)
    U_ = U[:, :r]
    D_ = np.diag(d[:r])
    V_ = V[:r]
    basis = U_
    vec = D_.dot(V_)
    print S.shape
    print basis.shape
    print vec.shape

    return basis, vec
    
B, V = find_S_vector_decomposition(S)


# In[106]:

class ListTable(list):
    """ Overridden list class which takes a 2-dimensional list of 
        the form [[1,2,3],[4,5,6]], and renders an HTML Table in 
        IPython Notebook. """
    
    def _repr_html_(self):
        html = ["<table>"]
        for row in self:
            html.append("<tr>")
            
            for col in row:
                html.append("<td>{0}</td>".format(col))
            
            html.append("</tr>")
        html.append("</table>")
        return ''.join(html)
    
def build_comparisane_table():
    table = ListTable()

    words = ['wood', 'woods', 'man', 'woman', 'dog', 'cat', 'picked', 'laid', 'taken', 'given', 'missed', 'signed', 'nice', 'good', 'bad']
    vecdict = dict([(w, V[:, D.index(w)]) for w in words])
    table.append([' '] + words)

    for w1 in words:
        row = []
        for w2 in words:
            v1 = vecdict[w1]
            v2 = vecdict[w2]
            row.append(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)))
        row = [w1] + ['%.4f' % row[id] if row[id] < 0.07 else '<b>%.4f</b>'%row[id] for id in range(len(row))]
        table.append(row)

    return table

build_comparisane_table()

