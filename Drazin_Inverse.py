import numpy as np
from scipy import linalg as la #used in index() helper function
from numpy.linalg import matrix_power as linAlgMP #used in is_drazin()
from scipy.sparse.csgraph import laplacian as LAP #used in effective resistance
import csv #used in LinkPredictor constructor
#to be able to see whole numpy array : see below (potentially for Link Predictor)
#https://stackoverflow.com/questions/1987694/how-to-print-the-full-numpy-array-without-truncation

def index(A, tol=1e-5):
    """Compute the index of the matrix A, referring to the
        minimum number k such that the Nullspace of (A^k) = N(A^k+1)

    Parameters:
        A ((n,n) ndarray): An nxn matrix.

    Returns:
        k (int): The index of A.
    """

    # test for non-singularity
    if not np.isclose(la.det(A), 0):
        return 0

    n = len(A)
    k = 1
    Ak = A.copy()
    while k <= n:
        r1 = np.linalg.matrix_rank(Ak)
        r2 = np.linalg.matrix_rank(np.dot(A,Ak))
        if r1 == r2:
            return k
        Ak = np.dot(A,Ak)
        k += 1

    return k

def is_drazin(A, Ad, k):
    """Verify that a matrix Ad is the Drazin inverse of A.

    Parameters:
        A ((n,n) ndarray): An nxn matrix.
        Ad ((n,n) ndarray): A candidate for the Drazin inverse of A.
        k (int): The index of A.

    Returns:
        (bool) True of Ad is the Drazin inverse of A, False otherwise.
    """
    #np.allclose returns boolean of whether two arrays (parameters) are equivalent
    #check first condition, if true then continue, else return false
    if np.allclose(A @ Ad, Ad @ A):
    # if np.allclose(A * Ad, Ad * A): #this shouldn't work
      pass
    else:
      return False
    #check second condition, if true then continue, else return false
    if np.allclose(linAlgMP(A,k+1) @ Ad, linAlgMP(A,k) ):
      pass
    else:
      return False
    #this final test case will only evaluate if it passes all previous cases
    #return the final condition whether it's True or False
    return np.allclose(Ad @ A @ Ad, Ad)

def test_is_drazin():
    """ test the is_drazin with 2 test cases"""
    A_k = 1
    A = np.array([
      [1,3,0,0],
      [0,1,3,0],
      [0,0,1,3],
      [0,0,0,0]
      ])
    Ad = np.array ([
      [1,-3,9,81],
      [0,1,-3,-18],
      [0,0,1,3],
      [0,0,0,0]
      ])
    B_k = 3
    B = np.array([
      [1,1,3],
      [5,2,6],
      [-2,-1,-3]
      ])
    Bd = np.array([
      [0,0,0],
      [0,0,0],
      [0,0,0]
    ])
    assert is_drazin(A,Ad,A_k),"failed A test case"
    assert is_drazin(B,Bd,B_k), "failed B test case"
    print("finished with and passed the is_drazin test cases")

def drazin_inverse(A, tol=1e-4):
    """Compute the Drazin inverse of A.

        Parameters:
            A ((n,n) ndarray): An nxn matrix.

        Returns:
            ((n,n) ndarray) The Drazin inverse of A.
        """
    n = np.shape(A)[0]
    f = lambda x: abs(x) > tol
    g = lambda x: abs(x) < tol
    #sort the schur decomposition
    Q1, S, k1 = la.schur(A,sort=f)
    Q2, T, k2 = la.schur(A,sort=g)
    U = np.hstack((S[:,:k1],T[:,:n-k1]))
    U_inv = la.inv(U)
    V = U_inv @ A @ U
    #initialize Z as matrix of floats
    Z = np.zeros((n,n),dtype=np.float64)
    if k1 != 0:
        M_inv = la.inv(V[:k1,:k1])
        Z[:k1,:k1] = M_inv
    return U @ Z @ U_inv

def test_drazin_inverse():
    """ test the drazin_inverse function """
    A_k = 1
    A = np.array([
      [1,3,0,0],
      [0,1,3,0],
      [0,0,1,3],
      [0,0,0,0]
      ])
    Ad = np.array ([
      [1,-3,9,81],
      [0,1,-3,-18],
      [0,0,1,3],
      [0,0,0,0]
      ])
    B_k = 3
    B = np.array([
      [1,1,3],
      [5,2,6],
      [-2,-1,-3]
      ])
    Bd = np.array([
      [0,0,0],
      [0,0,0],
      [0,0,0]
    ])
    AD = drazin_inverse(A)
    BD = drazin_inverse(B)
    assert is_drazin(A,AD,A_k),"failed A test case"
    assert is_drazin(B,BD,B_k), "failed B test case"
    assert np.allclose(Ad,AD), "Ad does not equals AD"
    assert np.allclose(Bd,BD), "Bd does not equals BD"
    assert index(A) == A_k, "index function failed w/ A"
    assert index(B) == B_k, "index function failed w/ B"
    print("finished with and passed the is_drazin test cases")

def effective_resistance(A):
    """Compute the effective resistance for each node in a graph.

        Parameters:
            A ((n,n) ndarray): The adjacency matrix of an undirected graph.

        Returns:
            ((n,n) ndarray) The matrix where the ijth entry is the effective
            resistance from node i to node j.

        """
    n = np.shape(A)[0]
    #zeros_like must be of type float or produces wrong output
    R = np.zeros_like(A,dtype=np.float64)
    L = LAP(A)
    I = np.eye(n)
    for j in range(n):
        # Lt is for L tilde
        Lt = np.copy(L)
        #replace jth row of Lt with jth row of Identity matrix
        Lt[j] = I[j]
        LtD = drazin_inverse(Lt)
        #replace column of R with diagonal of Drazin of Lt
        R[:,j] = LtD.diagonal()
    #adjust diagonal so that effective resistance from one node to itself is zero
    np.fill_diagonal(R,0.)
    return R

def resistance_test():
    """ Test effective_resistance function"""
    tol = 1e-7
    #A represents a straight linked node graph, test effective resistance from node a to d
    A = np.array([
        [0,1,0,0],
        [1,0,1,0],
        [0,1,0,1],
        [0,0,1,0]
    ])
    #effective resistance of matrix A from a node to d node is called A_a_d
    A_a_d = 3.
    #AR is effective resistance matrix of A
    AR = effective_resistance(A)
    # print(A,"\n above is matrix A, below is R\n",AR)
    # print(AR[0][3]) #note that it's a float close to 3. but not exactly
    assert abs(AR[0][3] - A_a_d) < tol, "test case A failed "
    #B is a simple graph with two nodes, only one edge
    B = np.array([
        [0,1],
        [1,0]
    ])
    B_a_b = 1.
    BR = effective_resistance(B)
    # print(B,"\n above is matrix A, below is R\n",BR)
    assert abs(BR[0][1] - B_a_b) < tol, "test case B failed"

    #C is a triangular graph, 3 nodes each having an edge to each other node
    C = np.array([
        [0,1,1],
        [1,0,1],
        [1,1,0]
    ])
    C_a_b = 2/3
    CR = effective_resistance(C)
    assert abs(CR[0][1] - C_a_b) < tol, "test case C failed"

    #graph 2ith 2 nodes that have 3 edges to the other node
    D = np.array([
        [0,3],
        [3,0]
    ])
    D_a_b = 1/3
    DR = effective_resistance(D)
    assert abs(DR[0][1] - D_a_b) < tol, "test case D failed "

    #E is a graph with 2 nodes, with two edges, like a circle
    E = np.array([
        [0,2],
        [2,0]
    ])
    E_a_b = 1/2
    ER = effective_resistance(E)
    assert abs(ER[0][1] - E_a_b) < tol, "test case E failed "

    #graph with 2 nodes, with 4 edges
    F = np.array([
        [0,4],
        [4,0]
    ])
    F_a_b = 1/4
    FR = effective_resistance(F)
    assert abs(FR[0][1] - F_a_b) < tol, "test case F failed"

    print("finished all test cases for effective_resistance function ")

class LinkPredictor:
    """Predict links between nodes of a network."""

    def __init__(self, filename='social_network.csv'):
        """Create the effective resistance matrix by constructing
            an adjacency matrix.

            Parameters:
                filename (str): The name of a file containing graph data.
            """
        #have names as a set to avoid repeats, then cast as list
        names = set()
        #indices will be used in constructing the adjacency matrix
        indices = dict()
        with open(filename, 'r') as infile:
            data = list(csv.reader(infile))
        #get a set of all people
        for line in data:
            #there will be two people on each line
            for person in line:
                names.add(person)

        #store list of names
        list_names = list(names)
        self.names = list_names

        #construct adjacency matrix
        n = len(list_names)
        adj = np.zeros((n,n))
        for line in data:
            a,b = line[0], line[1]
            ind1 = list_names.index(a)
            ind2 = list_names.index(b)
            # += 1 allows for multiple connection between two people
            #which isn't realistic on facebook but is realistic for circuits

            #the adjacency should be symmetric
            adj[ind1][ind2] += 1
            adj[ind2][ind1] += 1
        self.adjacency = adj
        self.resistance = effective_resistance(adj)

    def predict_link(self, node=None):
        """Predict the next link, either for the whole graph or fo 1r a
            particular node.

            Parameters:
                node (str): The name of a node in the network.

            Returns:
                node1, node2 (str): The names of the next nodes to be linked.
                    Returned if node is None.
                node1 (str): The name of the next node to be linked to 'node'.
                    Returned if node is not None.

            Raises:
                ValueError: If node is not in the graph.
            """
        #we assume that the adjacency matrix will only have 1's, and 0's

        #in order to return name which should be connected to node next
        #first, subtract one so that adjacency has values where nodes have no connection
        #multiply by -1 to have positive values to find minimium
        #multiply by resistance to find which node are most likely to be connected
        modified = (self.adjacency - np.ones_like(self.adjacency,np.float64)) * -1 * self.resistance
        # modified = (self.adjacency - 1) * -1 * self.resistance #NOTE: this is what I had before 
        np.fill_diagonal(modified,0.)
        if node is None:
            #return a tuple with the names of nodes between the next link should occur
            #get minimum resitance value
            minval = np.min(modified[np.nonzero(modified)])
            # return minval
            #loc for location
            loc = np.where(modified == minval)
            index1 = loc[0][0]
            index2 = loc[1][0]
            #get names of nearest neighbors
            name1 = self.names[index1]
            name2 = self.names[index2]
            return (name1,name2)
        else:
            if node not in self.names:
                raise ValueError("\"{}\" is not in network".format(node))
            else:
                #get the index of the node
                row = self.names.index(node)
                #get minimum resitance value
                minval = np.min(modified[:,row][np.nonzero(modified[:,row])])
                #loc for location
                loc = np.where(modified[:,row] == minval)
                index1 = loc[0][0]
                #get name of nearest neighbor
                name = self.names[index1]
                return name
        #this will not be evaluated because of preceding return statements
        raise NotImplementedError("predict_link method incomplete")

    def add_link(self, node1, node2):
        """Add a link to the graph between node 1 and node 2 by updating the
            adjacency matrix and the effective resistance matrix.

            Parameters:
                node1 (str): The name of a node in the network.
                node2 (str): The name of a node in the network.

            Raises:
                ValueError: If either node1 or node2 is not in the graph.
            """
        if node1 not in self.names:
            raise ValueError("{} is not in network".format(node1))
        if node2 not in self.names:
            raise ValueError("{} is not in network".format(node2))
        ind1 = self.names.index(node1)
        ind2 = self.names.index(node2)
        self.adjacency[ind1][ind2] += 1
        self.adjacency[ind2][ind1] += 1
        #the return isn't necessary but is good practice
        return
        # raise NotImplementedError("Add_link method incomplete")

def test_class():
    """ test the LinkPredictor """
    myLinks = LinkPredictor()
    # """

    l = myLinks.predict_link(node=None) #I could have left blank for default
    # print(l) #should say "Emily" and "Oliver"
    assert l == ("Emily", "Oliver") or l == ("Oliver","Emily"), "failed node=None case "

    o = myLinks.predict_link("Melanie")
    # print("Output is \'{}\' but should say \"Carol\" ".format(o) )
    assert o == 'Carol', "Failed Melanie Case"

    try:
        a = myLinks.predict_link("Not a Name")
    except ValueError:
        pass
        # print("correctly raised ValueError")

    a1 = myLinks.predict_link("Alan")
    # print("Output is \'{}\' but should say \"Sonia\" ".format(a1) )
    assert a1 == "Sonia", "failed first Alan case"
    myLinks.add_link("Alan","Sonia")

    a2 = myLinks.predict_link("Alan")
    # print("Output is \'{}\' but should say \"Piers\" ".format(a2) )
    assert a2 == "Piers", "failed second Alan case"
    myLinks.add_link("Alan","Piers")

    a3 = myLinks.predict_link("Alan")
    # print("Output is \'{}\' but should say \"Abigail\" ".format(a3) )
    assert a3 == "Abigail", "failed third Alan case"
    myLinks.add_link("Alan","Abigail")

    a4 = myLinks.predict_link("Alan")
    # print("The result from fourth Alan case is {}".format(a4))

    #add a link between some nodes, first verify there isn't a connection
    a = "Eric"
    b = "Jake"
    ind1 = myLinks.names.index(a)
    ind2 = myLinks.names.index(b)
    # print(myLinks.adjacency[ind1][ind2])
    assert myLinks.adjacency[ind1][ind2] == 0.0, "there is a connection between {} & {} Already".format(a,b)
    #then add the connection
    myLinks.add_link(a,b)
    # print(myLinks.adjacency[ind1][ind2])
    assert myLinks.adjacency[ind1][ind2] == 1, "the connection didn't add to the adjacency for {} and {}".format(a,b)

    # """
    #test cases from the driver
    #nodes 18, 4, 33
    a5 = myLinks.predict_link("John")
    print("Output from predict link for John should be Stephanie but is:",a5)
    myLinks.add_link(a5,"John")
    b1 = myLinks.predict_link("John")
    print("Output from predict link for John (called a second time) could be Stephanie but is:",b1)

    #nodes 5, 2, 3 (equal opportunity)
    a6 = myLinks.predict_link("Carol")
    print("Output from predict link for Carol should be Abigail but is:",a6)
    myLinks.add_link(a6,"Carol")
    b2 = myLinks.predict_link("Carol")
    print("Output from predict link for Carol (second time) could be oliver? but is:",b2)
    # """

    print("\nfinished test cases for LinkPredictor Class ")

if __name__ == "__main__":
    pass
    # test_is_drazin()
    # test_drazin_inverse()
    # resistance_test()
    # myLinks = LinkPredictor()
    # print(myLinks.adjacency)
    # print("\n",myLinks.resistance)
    test_class()
    # print("don't forget to uncomment test cases in test_class()")

    """ cool test case
    A = np.array([
        [0,1,0,1,1,1],
        [1,0,1,0,0,0],
        [0,1,0,1,1,1],
        [1,0,1,0,0,0],
        [1,0,1,0,0,0],
        [1,0,1,0,0,0]
    ])
    A_names = ["joe","billy","tommy","jessica","hannah","MJ"]
    """
