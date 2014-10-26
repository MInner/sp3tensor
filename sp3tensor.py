import time

import numpy as np
from scipy.sparse import csc_matrix, coo_matrix
import scipy as sp

import matplotlib.pyplot as plt

def layer_print(T):
    for i in range(T.shape[2]):
        print T[:, :, i]

def dense_convolution(A, B):
    return naive_unfold1(A).dot(naive_unfold2(B).T)
    
def dense_tensmatr(A, D):
    I, J, K = A.shape
    M = D.shape[1]
    return A.transpose(2, 0, 1).reshape(K, I*J).dot( np.kron(np.eye(A.shape[0]), D) ).T.reshape(I, M, K)

def naive_unfold1(A):
    return A.reshape(A.shape[0], A.shape[2]*A.shape[1])

def naive_unfold2(A):
    I, J, K = A.shape
    return A.transpose(0, 2, 1).reshape(I*K, J).T


# ### Sparce densor implemenation

class sp3tensor:
    """ Sparce tensor that stores data in sparce array format:
        data - values
        indexes - i-th of non-zero data
        , e.g. [0 0 0 0 1 0 0 2] -> val: [1 2], ind: [4, 7].
        
        Size: I, J, K
        Number of elements: N
        
        Complexisites:
         get - O(N) (must be log N) - need to replace lsits with heaps
         set - O(N) (must be log N)
         tofold_* - O(N)
         """
    
    def __init__(self, values, indexes, shape):
        if len(shape) != 3:
            raise NotImplementedError, "Only 3-dim tensors are supported"
            
        if len(values) != len(indexes):
            raise ValueError, "Values and Indexes must be of same size"
        
        self.shape = shape
        self.__values = values
        self.__indexes = indexes
        
    def index_from3d(self, i, j, k):
        return i*self.shape[1]*self.shape[2] + j*self.shape[2] + k
        
    def __setitem__(self, key, item):
        self.__check_bounds(key)
            
        try:
            ind = self.index_from3d(*key)
            internal_ind = self.__indexes.index(ind) # must be replaced with log n data structure !
            self.__values[internal_ind] = item
        except ValueError:
            self.__indexes.append(ind)
            self.__values.append(item)
            
    def __getitem__(self, key):
        self.__check_bounds(key)
        try:
            ind = self.index_from3d(*key)
            internal_ind = self.__indexes.index(ind)  # must be replaced with log n data structure !
            return self.__values[internal_ind]
        except ValueError:
            return 0
    
    def tolayers(self):
        """ Returns list of flat k-mode layers"""
        ls = []
        i = np.array(self.__indexes)
        v = np.array(self.__values)
        belogs_to = i % self.shape[2]
        for k in range(self.shape[2]):
            loc_i = i[belogs_to == k]
            loc_v = v[belogs_to == k]
            loc_map = (loc_i-k)/self.shape[2]
            row = loc_map / self.shape[1]
            col = loc_map - row*self.shape[1]
            ls.append(coo_matrix((loc_v, (row, col)), shape=(self.shape[:2])))
        return ls
        
    @property
    def nnz(self):
        return len(self.__values)
    
    @property
    def T(self):
        i = np.array(self.__indexes)
        # fold-3 representatinon
        col = i/self.shape[2]
        row = np.mod(i, self.shape[2])
        t_row = row # rows - the tubes themselves stays same (transponated)
        t_col = np.mod(col, self.shape[1])*self.shape[0] + col/self.shape[1]
        new_i = self.shape[2]*t_col + t_row
        return sp3tensor(self.__values, new_i, shape = (self.shape[1], self.shape[0], self.shape[2]))
        
    @property
    def data(self):
        return self.__values
    
    def __check_bounds(self, key):
        if len(key) != len(self.shape):
            raise IndexError, "Key is %s and shape of array is %s - on match" % (key, self.shape) 
        else:
            for k, s in zip(key, self.shape):
                if k >= s or k < 0:
                    raise IndexError, "Key %s is out of bounds %s" % (key, self.shape)
                    
                    
    def tofold_1(self):
        i = np.array(self.__indexes)
        rows = i / ( self.shape[1] * self.shape[2] )
        cols = i - rows*self.shape[1] * self.shape[2]
        return coo_matrix((self.__values, (rows, cols)), shape=(self.shape[0], self.shape[1]*self.shape[2])).tocsc()
    
    def tofold_2(self):
        i = np.array(self.__indexes)
        I, J, K = self.shape
        block_id = i / (K*J) # vettical blicks of width K
        inner_i = i - block_id*K*J
        inner_row = inner_i/K
        inner_col = np.mod(inner_i, K)
        row = inner_row
        col = block_id*K + inner_col
        return coo_matrix((self.__values, (row, col)), shape=(J, I*K)).tocsc()
    
#     def tofold_3(self):
#         i = np.array(self.__indexes)
#         rows = np.mod(i, self.shape[2])
#         cols = i / self.shape[2]
#         return coo_matrix((self.__values, (rows, cols)), shape=(self.shape[2], self.shape[0]*self.shape[1])).tocsc()
    
    def tenmat(self, M):
        raise DeprecationWarning, 'Slow version! Don\' use it! - use tenmat3 instead'
        if self.shape[1] != M.shape[0]:
            raise ValueError, "Cant tenmat of sizes: %s, %s" % (self.shape, M.shape)
        c = self.tofold_3().dot( sp.sparse.kron(sp.sparse.eye(self.shape[0]), M) ).tocoo()
        i = c.row + c.shape[0]*c.col
        v = c.data
        return sp3tensor(v, i, (self.shape[0], M.shape[1], self.shape[2]))
    
    def tenmat2(self, M):
        raise DeprecationWarning, 'Slow version! Don\' use it! - use tenmat3 instead'
        """ Should work faster for smaller K"""
        if self.shape[1] != M.shape[0]:
            raise ValueError, "Cant tenmat of sizes: %s, %s" % (self.shape, M.shape)
        ls = self.tolayers()
        return sp3tensor.fromlayers([l.dot(M) for l in ls])
    
    def tenmat3(self, M, sparce = False):
        """ Should work faster for smaller K - with hstack"""
        if len(M.shape) != 2:
            raise ValueError, "You tenmat tensor by matrix only!" 
        
        if self.shape[1] != M.shape[0]:
            raise ValueError, "Cant tenmat of sizes: %s, %s" % (self.shape, M.shape)
        rls = [l.dot(M) for l in self.tolayers()]
        I, J = rls[0].shape
        K = len(rls)
        dense = np.hstack([x.reshape((I*J, 1)) for x in rls]).reshape(I, J, K)
        if sparce:
            return sp3tensor.fromtensor( dense )
        else:
            return dense
        
    def tenten(self, T):
        if self.shape[1] != T.shape[0] or self.shape[2] != T.shape[2]:
            raise ValueError, "Cant tenten of sizes: %s, %s" % (self.shape, T.shape)
        
        if T.__class__ == sp3tensor:
            return self.tofold_1().dot( T.tofold_2().transpose() )
        if T.__class__ == np.ndarray:
            return self.tofold_1().dot( naive_unfold2(T).T )
            
    @classmethod
    def zeros(cls, shape):
        self = cls([], [], shape)
        return self
    
    @classmethod
    def fromtensor(cls, T):
        if len(T.shape) != 3:
            raise NotImplementedError, "Only 3-dim tensors are supported"
            
        v = T.reshape(np.prod(T.shape))
        i = np.arange(v.size)
        
        to_del = i[v == 0]
        i = np.delete(i, to_del)
        v = np.delete(v, to_del)
        self = cls(v.tolist(), i.tolist(), T.shape)
        return self
    
    @classmethod
    def fromlayers(cls, ls):
        if len(ls) == 0:
            raise ValueError, "Layers list must be non-empty"
            

        if type(ls[0]) == coo_matrix:
            new = cls.zeros((ls[0].shape[0], ls[0].shape[1], len(ls)))
            vals = []
            ids = []
            for k, l in enumerate(ls):
                vals.extend(l.data)
                ids.extend(new.index_from3d(l.row, l.col, np.ones(len(l.row), dtype = int)*k))
            return cls(vals, ids, (ls[0].shape[0], ls[0].shape[1], len(ls)))
        
#         if type(ls[0]) == np.ndarray:
#             new = cls.zeros((ls[0].shape[0], ls[0].shape[1], len(ls)))
#             vals = []
#             ids = []
#             ar = np.arange(np.prod(ls[0].shape))
#             for k, l in enumerate(ls):
#                 vals.extend(l.reshape(np.prod(l.shape)))
#                 rows = ar/l.shape[1]
#                 cols = np.mod(ar, l.shape[1])
#                 ids.extend(new.index_from3d(rows, cols, np.ones(len(rows), dtype = int)*k))
#             return cls(vals, ids, (ls[0].shape[0], ls[0].shape[1], len(ls)))
        I, J = ls[0].shape
        K = len(ls)
        dense = np.hstack([x.reshape((I*J, 1)) for x in ls])
        dense = np.asarray(dense).reshape(I, J, K)
        return cls.fromtensor(dense)
        
    def todense(self):
        toret = np.zeros(np.prod(self.shape))
        toret[self.__indexes] = self.__values
        return toret.reshape(self.shape)
    
    def __repr__(self):
        return "<sp3tensor \n  val:%s , \n  ind:%s>" % (str(self.__values), str(self.__indexes))


def test_leyer_decomposition():
    for i in range(100):
        I = 2
        J = 3
        L = 4
        A = np.arange(I*J*L).reshape(I, J, L)
        A_ = sp3tensor.fromtensor(A)
        if np.linalg.norm(sp3tensor.fromlayers(A_.tolayers()).todense() - A) > 0.0001:
            print 'Test failed!"'
            break
    print 'All tests passed'

def test_unfold1():
    for i in range(100):
        I = np.random.randint(5, 100)
        J = np.random.randint(5, 100)
        K = np.random.randint(1, 5)

        A = np.zeros((I, J, K))

        A[np.random.randint(0, I, I), np.random.randint(0, J, I), np.random.randint(0, K, I)] = np.random.rand(I)
        A_ = sp3tensor.fromtensor(A)

        if (np.linalg.norm(naive_unfold1(A) - A_.tofold_1())) > 0.0001:
            print 'Test failed!'
            break
    print 'All tests are passed'

def test_unfold2():
    for i in range(100):
        I = np.random.randint(5, 100)
        J = np.random.randint(5, 100)
        K = np.random.randint(1, 5)

        A = np.zeros((I, J, K))

        A[np.random.randint(0, I, I), np.random.randint(0, J, I), np.random.randint(0, K, I)] = np.random.rand(I)
        A_ = sp3tensor.fromtensor(A)

        if (np.linalg.norm(naive_unfold2(A) - A_.tofold_2())) > 0.0001:
            print 'Test failed!'
            break
    print 'All tests are passed'

# ### Tensor-tensor multiplication test

def test_tenten():
    for i in range(100):
        I = np.random.randint(5, 100)
        J = np.random.randint(5, 100)
        K = np.random.randint(1, 5)

        A = np.zeros((I, J, K))
        B = np.zeros((J, I, K))

        A[np.random.randint(0, I, I), np.random.randint(0, J, I), np.random.randint(0, K, I)] = np.random.rand(I)
        A_ = sp3tensor.fromtensor(A)

        B[np.random.randint(0, J, I), np.random.randint(0, I, I), np.random.randint(0, K, I)] = np.random.rand(I)
        B_ = sp3tensor.fromtensor(B)
        
        if np.linalg.norm(A_.tenten(B_) - dense_convolution(A, B)) > 0:
            print 'Test failed!'
            break
            
    print 'All tentendot test are passed!'
        

# ### Tensor-matrix multiplication test

def test_tenmat():
    for i in range(100):
        I = np.random.randint(5, 100)
        J = np.random.randint(5, 100)
        K = np.random.randint(1, 5)
        M = np.random.randint(5, 100)
        
        A = np.zeros((I, J, K))
        D = np.random.randint(0, 10, J*M).reshape(J, M)
        
        A[np.random.randint(0, I, I), np.random.randint(0, J, I), np.random.randint(0, K, I)] = np.random.rand(I)
        A_ = sp3tensor.fromtensor(A)
        
        if np.linalg.norm(A_.tenmat(D).todense() - dense_tensmatr(A, D)) > 0.00001:
            print 'Test failed!'
            break
            
    print 'All test are passed!'
    


def test_tenmat3():
    for i in range(100):
        I = np.random.randint(5, 100)
        J = np.random.randint(5, 100)
        K = np.random.randint(1, 5)
        M = np.random.randint(5, 100)
        
        A = np.zeros((I, J, K))
        D = np.random.randint(0, 10, J*M).reshape(J, M)
        
        A[np.random.randint(0, I, I), np.random.randint(0, J, I), np.random.randint(0, K, I)] = np.random.rand(I)
        A_ = sp3tensor.fromtensor(A)
        
        if np.linalg.norm(A_.tenmat3(D).todense() - dense_tensmatr(A, D)) > 0.00001:
            print 'Test failed!'
            break
            
    print 'All test are passed!'
    
# ### Tensor T test

def transpose_test():
    for i in range(100):
        I = np.random.randint(5, 10)
        J = np.random.randint(5, 10)
        K = np.random.randint(1, 5)

        A = np.zeros((I, J, K))
        A[np.random.randint(0, I, 3*I), np.random.randint(0, J, 3*I), np.random.randint(0, K, 3*I)] = np.random.rand(3*I)

        A_ = sp3tensor.fromtensor(A)
        
        if np.linalg.norm(A_.T.todense() - A.transpose(1, 0, 2)) > 0:
            print 'Test failed!'
            break
            
    print 'All tests have passed!'

# ### Checking speed in compariasne with dense tensors

def speedcheck():
    n_range = range(20, 500, 10)
    
    tenmat_N = 120
    n_range_tenmat = range(20, tenmat_N, 10)
    
    dbt = [] # dense build time
    sbt = [] # sparce build time
    ddt = [] # dense dot time
    sdt = [] # sparce dot time
    dmt = [] # dense mat-prod time
    smt = [] # sparce mat-prod time
    smt2 = [] # sparce mat-prod time (second way)
    
    for N in n_range:
        if N%50 == 0:
            print 'Size is', N
        L = 2
        
        D = np.random.rand(N, N)

        A = np.zeros((N, N, L))
        A[np.random.randint(0, N, 2*N), np.random.randint(0, N, 2*N), np.random.randint(0, L, 2*N)] = np.random.rand(2*N)
        A_ = sp3tensor.fromtensor(A)
        
        s = time.clock()
        B = np.zeros((N, N, L))
        for i, j, k in zip(np.random.randint(0, N, 2*N), np.random.randint(0, N, 2*N), np.random.randint(0, L, 2*N)):
            B[i, j, k] += np.random.randint(0, 2*N)
        dbt.append( (time.clock() - s) )
        
        s = time.clock()
        B_ = sp3tensor.zeros((N, N, L))
        for i, j, k in zip(np.random.randint(0, N, 2*N), np.random.randint(0, N, 2*N), np.random.randint(0, L, 2*N)):
            B_[i, j, k] += np.random.randint(0, N)
        sbt.append( (time.clock() - s) )

        s = time.clock()
        for i in range(10):
            A_.tenten(B_)
        sdt.append( (time.clock() - s) )
        
        s = time.clock()
        for i in range(10):
            A.reshape(N, L*N).dot(B.transpose(0, 2, 1).reshape(L*N, N))    
        ddt.append( (time.clock() - s) )
        
        if N < tenmat_N:
        
            s = time.clock()
            for i in range(10):
                A_.tenmat(D)
            smt.append( (time.clock() - s) )
            
            s = time.clock()
            for i in range(10):
                A_.tenmat2(D)
            smt2.append( time.clock() - s )

            s = time.clock()
            for i in range(10):
                dense_tensmatr(A, D)
            dmt.append( time.clock() - s )

    plt.plot(n_range, dbt, label = 'Dense')
    plt.plot(n_range, sbt, label = 'Sparce')
    plt.title('Build time (assignment speed)')
    plt.legend()
    plt.show()
    
    plt.plot(n_range, ddt, label = 'Dense')
    plt.plot(n_range, sdt, label = 'Sparce')
    plt.title('Special Tensor-Tensor product time (multiplication speed)')
    plt.legend()
    plt.show()
    
    plt.plot(n_range_tenmat, dmt, label = 'Dense')
    plt.plot(n_range_tenmat, smt, label = 'Sparce')
    plt.plot(n_range_tenmat, smt2, label = 'Sparce-2')
    plt.title('Special Tensor-Matrix (multiplication speed)')
    plt.legend()
    plt.show()
        
# 

# ### Testing multiplications on huge numbers (thousands)

def test_huge_numbers(repeates = 5):
    n_range = range(1000, 6000, 1000)
    data = []
    L = 2
    for N in n_range:
        print N
        A = np.zeros((N, N, L))
        A[np.random.randint(0, N, 2*N), np.random.randint(0, N, 2*N), np.random.randint(0, L, 2*N)] = np.random.rand(2*N)
        A_ = sp3tensor.fromtensor(A)
        
        D = np.random.rand(N, N)
        
        s = time.clock()
        for i in range(repeates):
            A_.tenmat2(D)
        t1 = time.clock() - s
        
        s = time.clock()
        for i in range(repeates):
            A_.tenmat3(D)
        t2 = time.clock() - s
        
        s = time.clock()
        for i in range(repeates):
            A_.tenmat3(D, sparce = True)
        t3 = time.clock() - s
        
        s = time.clock()
        for i in range(repeates):
            A_.tenten(A_)
        t4 = time.clock() - s
        
        data.append( (t1/repeates, t2/repeates, t3/repeates,  t4/repeates) )
        
    plt.plot(n_range, zip(*data)[0], label = 'tenmat2')
    plt.plot(n_range, zip(*data)[1], label = 'tenmat3')
    plt.plot(n_range, zip(*data)[2], label = 'tenmat3-sparce')
    plt.title('Martix-Tensor Special Prod Time on huge N')
    plt.legend()
    plt.show()
    plt.plot(n_range, zip(*data)[3], label = 'tenten')
    plt.title('Tensor-Tensor Special Prod Time on huge N')
    plt.show()

def all_test():
    test_leyer_decomposition()
    test_unfold1()
    test_unfold2()
    test_tenten()
    test_tenmat()
    test_tenmat3()
    transpose_test()
    # speedcheck()
    # test_huge_numbers(repeates = 3)
