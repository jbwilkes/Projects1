import numpy as np
import random
from scipy import linalg as la
import pdb

def random_chain(n):
    """Create and return a transition matrix for a random Markov chain with
    'n' states. This should be stored as an nxn NumPy array.

    plan: create a random matrix, use scipy to normalize the columns, then return it
    """
    #create a random matrix
    A = np.random.random((n,n))
    #construct it so the columns add up to one
    B = np.sum(A,axis = 0)
    # the difficulty is getting it to broadcast appropriately
    # website of reference - only minorly helpful - https://stackoverflow.com/questions/19602187/numpy-divide-each-row-by-a-vector-element

    return A / B

def forecast(n):
    """Forecast n next days of weather given that today is hot."""
    transition = np.array([[0.7, 0.6], [0.3, 0.4]])
    weather = []
    previous_conditions = 0
    # Sample from a binomial distribution to choose a new state.
    #change the weather based upon previous day's conditions
    for _ in range(n):
        #if previous day was warm
        if previous_conditions == 0:
            weather.append(np.random.binomial(1, transition[1, 0]))
        #if the previous day was cold
        else:
            weather.append(np.random.binomial(1, transition[1, 1]))
        previous_conditions = weather[-1]

    return weather

def four_state_forecast(days):
    """Run a simulation for the weather over the specified number of days,
    with mild as the starting state, using the four-state Markov chain.
    Return a list containing the day-by-day results, not including the
    starting day.

    Examples:
        >>> four_state_forecast(3)
        [0, 1, 3]
        >>> four_state_forecast(5)
        [2, 1, 2, 1, 1]
    """
    four_state = np.array(
    [
        [0.5, 0.3, 0.1, 0  ],
        [0.3, 0.3, 0.3, 0.3],
        [0.2, 0.3, 0.4, 0.5],
        [0,   0.1, 0.2, 0.2]
    ])

    weather = []
    previous_conditions = 1
    for _ in range(days):
        if previous_conditions == 0:
            weather.append(np.argmax(np.random.multinomial(1, four_state[:,0])))
        elif previous_conditions == 1:
            weather.append(np.argmax(np.random.multinomial(1, four_state[:,1])))
        elif previous_conditions == 2:
            weather.append(np.argmax(np.random.multinomial(1, four_state[:,2])))
        else:
            weather.append(np.argmax(np.random.multinomial(1, four_state[:,3])))
        previous_conditions = weather[-1]

    return weather

def steady_state(A, tol=1e-12, N=400):
    """Compute the steady state of the transition matrix A.

    Inputs:
        A ((n,n) ndarray): A column-stochastic transition matrix.
        tol (float): The convergence tolerance.
        N (int): The maximum number of iterations to compute.

    Raises:
        ValueError: if the iteration does not converge within N steps.

    Returns:
        x ((n,) ndarray): The steady state distribution vector of A.
    """
    m,n = np.shape(A)
    #generate random state distribution vector
    xNaught = np.random.random(n)
    #the random vector elements must sum to one
    temp = np.sum(xNaught,axis = 0)
    x =  xNaught / temp
    k = 0
    #calculate limit of x as k approaches N
    while True:
        xPlus = A @ x
        #if the vector is less than the tolerance then return it
        if la.norm(xPlus - x) < tol:
            break
        else:
            x = xPlus
        if k >= N:
            raise ValueError("transition matrix does not converge")
        else:
            k+= 1
    return x

def steady_state_test():
    """ test the steady state function"""
    transition = np.array([[0.7, 0.6], [0.3, 0.4]])
    two_by_two_vector_limit = np.array([2/3,1/3])
    print(steady_state(transition),two_by_two_vector_limit, sep='\n')
    four_state = np.array(
    [
        [0.5, 0.3, 0.1, 0  ],
        [0.3, 0.3, 0.3, 0.3],
        [0.2, 0.3, 0.4, 0.5],
        [0,   0.1, 0.2, 0.2]
    ])


    # four_state_vector_limit = np.array([])

class SentenceGenerator(object):
    """Markov chain creator for simulating bad English.

    Attributes:
        (what attributes do you need to keep track of?)

    Example:
        >>> yoda = SentenceGenerator("Yoda.txt")
        >>> print(yoda.babble())
        The dark side of loss is a path as one with you.
    """
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        library = set()
        #read in the content
        with open(filename, 'r') as myfile:
            contents = myfile.readlines()
        #segment each line into sublists
        for i in range(len(contents)):
            contents[i] = contents[i].strip().split(' ')
            # contents[i] = contents[i].split(' ')
        #for each sublist, add each entry to the library set
        for j in range(len(contents)):
        # for j in range(3): #to see a small subset of the file
            l = len(contents[j])
            for k in range(l):
                library.add(contents[j][k])

        count = len(library)
        #initialize as a data member for use in the babble method
        self.count = count
        #the matrix maps the column word to the next word in the sentence to the row corresponding to the next word
        #the domain is the columns, mapping to the rows
        matrix = np.zeros((count+2,count+2))
        indices = {"$tart":0,"$top":count+1}
        #j represents the index
        j = 1
        #iterate through the set adding it to the dictionary with a index
        for i in library:
            indices.update({i:j})
            j += 1
            if j >= count + 2:
                raise ValueError("error with index")
        #intialize for use in the babble method
        self.indices = indices

        #for each sentence
        for sentence in contents:
            #connect $start with first word which has index 0 NOT ONE!!!!!!!
            matrix[indices.get(sentence[0])][0] += 1
            #for each word in the sentence map it to the next word and it's column increasing count by 1
            l = len(sentence)
            for k in range(l):
                # pdb.set_trace()
                #if it's the last word in the sentence it needs to map to stop
                if k == l-1:
                    matrix[-1][indices.get(sentence[k])] += 1
                else:
                    matrix[indices.get(sentence[k+1])][indices.get(sentence[k])] += 1
        #map stp to stop
        matrix[-1][-1] += 1

        #adjust each column so the sum is one
        temp = np.sum(matrix,axis = 0)
        matrix =  matrix / temp
        #initialize for use in the babble
        self.matrix = matrix

    def babble(self):
        """Begin at the start sate and use the strategy from
        four_state_forecast() to transition through the Markov chain.
        Keep track of the path through the chain and the corresponding words.
        When the stop state is reached, stop transitioning and terminate the
        sentence. Return the resulting sentence as a single string.
        """

        #string to be printed
        declaration = ""
        previous_conditions = 0
        #add a word until the final word is $top
        while previous_conditions < (self.count + 1):
            #the index of the random multinomial sample
            word_index = np.argmax(np.random.multinomial(1, self.matrix[:,previous_conditions]))
            #find the word for the given index
            word = list(self.indices.keys())[list(self.indices.values()).index(word_index)]
            if word != "$top":
                declaration += word + " "
            #set the previous condition with the probabilities for the next sample
            previous_conditions = word_index

        return declaration

if __name__ == "__main__":
    pass
    """
        favorites babbles:

        yoda
        mourn them do not, miss them do not

        trump:
        We don't want the country that's happened.
        Hey, I'm with our values?

        for more text processing - see this website
        https://www.gutenberg.org/
    """

    # print(random_chain(2))
    # print(forecast(3))
    # print(four_state_forecast(3))
    # steady_state_test()
    # file = SentenceGenerator("trump.txt")
    # file = SentenceGenerator("yoda.txt") #count is 687 unique words
    # file = SentenceGenerator("tswift1989.txt")
    # file = SentenceGenerator("sample.txt")
    # print(file.babble())
    #for trump the count is 11802 unique words, so thats a big matrix
