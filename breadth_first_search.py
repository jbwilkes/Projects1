# breadth_first_search.py
"""Volume 2: Breadth-First Search.
<Joseph Wilkes>
<sec 01>
<25 oct 18>
"""

import pdb
import time
# import numpy as np
import networkx as nx
from matplotlib import pyplot as plt

class Graph:
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a set of
    the corresponding node's neighbors.

    Attributes:
        d (dict): the adjacency dictionary of the graph.
    """
    def __init__(self, adjacency={}):
        """Store the adjacency dictionary as a class attribute"""
        self.d = dict(adjacency)

    def __str__(self):
        """String representation: a view of the adjacency dictionary."""
        return str(self.d)

    def add_node(self, n):
        """Add n to the graph (with no initial edges) if it is not already
        present.

        Parameters:
            n: the label for the new node.
        """
        #if n is not already a key, update it / add it to dictionary
        #Code: if n in self.d.keys(): #alternative way to write this find
        if n in self.d:
            pass
            #print("already in list")
        else:
            self.d.update({n : set()})

    def add_edge(self, u, v):
        """Add an edge between node u and node v. Also add u and v to the graph
        if they are not already present.

        Parameters:
            u: a node label.
            v: a node label.
        """
        #check to see if u, and v are not in dictionary already
        if u not in self.d.keys():
            self.add_node(u)
        if v not in self.d.keys():
            self.add_node(v)
        newEdgeSet = self.d[u]
        #add the edge v, to the pre-existing values of the u-key
        self.d[u] = newEdgeSet.union({v})
        newEdgeSet = self.d[v]
        #do the same for key-v
        self.d[v] = newEdgeSet.union({u})

    def remove_node(self, n):
        """Remove n from the graph, including all edges adjacent to it.

        Parameters:
            n: the label for the node to remove.

        Raises:
            KeyError: if n is not in the graph.
        """
        #throw an error if the node isn't in the dictionary
        if n not in self.d:
            raise KeyError(n,"is not in Graph")
        else:
            #remove all the edges to that node, then remove the node last
            #it'll throw an error if it's not in a given set
            list_OF_values = list(self.d.values())
            for i in list_OF_values:
                try:
                    i.remove(n)
                except KeyError:
                    pass
            self.d.pop(n)
            # print("removing node")

    def remove_edge(self, u, v):
        """Remove the edge between nodes u and v.

        Parameters:
            u: a node label.
            v: a node label.

        Raises:
            KeyError: if u or v are not in the graph, or if there is no
                edge between u and v.
        """
        if u not in self.d or v not in self.d:
            raise KeyError("one or both nodes are not in Graph")
        else:
            EdgeSet = self.d[u]
            EdgeSet.remove(v)
            EdgeSet = self.d[v]
            EdgeSet.remove(u)
            # print("removing edge")

    def traverse(self, source):
        """Traverse the graph with a breadth-first search until all nodes
        have been visited. Return the list of nodes in the order that they
        were visited.

        Parameters:
            source: the node to start the search at.

        Returns:
            (list): the nodes in order of visitation.

        Raises:
            KeyError: if the source node is not in the graph.
        """

        if source not in self.d :
            raise KeyError("source not is not in the graph")
        numNodes = len(self.d)
        #visited must be a list in order to keep the order as opposed to set
        visited = []
        queue = [source]

        #for each node - add all the neighbors to the queue
        while True:
            #list of neighbors of current node
            #reset the neighbor list just in case Node is a leaf
            neighborNodeSet = []

            #quite traversing once every node has been visited
            if len(visited) == numNodes:
                break
            #add current node to the visited
            if queue[0] not in visited:
                visited.append(queue[0])
                neighborNodeSet = self.d.get(queue[0])
            #remove it from the queue
            queue.pop(0)

            #add to the queue all the neighbors
            for i in neighborNodeSet:
                #only add to queue if node hasn't been visited yet
                if i not in visited:
                    queue.append(i)

        return visited

    def shortest_path(self, source, target):
        """
        Begin a BFS at the source node and proceed until the target is
        found. Return a list containing the nodes in the shortest path from
        the source to the target, including endoints.

        Parameters:
            source: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from source to target,
                including the endpoints.

        Raises:
            KeyError: if the source or target nodes are not in the graph.
        """
        if source not in self.d or target not in self.d:
            raise KeyError("one or both nodes are not in Graph")
        path = dict()
        visited = []
        queue = [source]
        marked = {source}
        #include endpoints in FINAL_PATH
        FINAL_PATH = []
        currentNode = source

        #### Perform the search
        while currentNode is not target:
            currentNode = queue.pop(0)
            neighborNodeSet = self.d.get(currentNode)
            #add neighbors of currentNodvisited.append(currentNode)e to Queue if they aren't in marked,
            #add the neighbors of currentNode to the marked list if they aren't in it already
            # pdb.set_trace()
            for i in neighborNodeSet:
                if i not in marked:
                    marked.add(i)
                    queue.append(i)
                    #when a element goes into marked, add it to the path dictionary backwards
                    # I could use add attribute for the path dictionary, not update, if I don't want it to overwrite which node arrived first to a given node
                    path.update({i:currentNode})

            #add currentNode to visited and reset currentNode
            visited.append(currentNode)
            # print(path)

        #### Evaluate the search using the dictionary
        currentNode = target
        while True:
            if currentNode == source:
                FINAL_PATH.append(currentNode)
                break
            FINAL_PATH.append(currentNode)
            currentNode = path.get(currentNode)

        #reverse the path
        return FINAL_PATH[::-1]

def test_graph():
    """ test the Graph class and methods"""
    myG = Graph()
    insert = ["a","b","a",7,10]
    for _ in insert:
        myG.add_node(_)
        # print(myG)
    for _ in insert:
        if _ != 7:
            myG.add_edge(_,7)
    # print(myG)
    myG.add_edge("a","c")
    myG.add_edge("c",10)
    myG.add_edge("b","c")
    myG.add_edge("a",10)

    print(myG)
    try:
        myG.remove_node("d")
    except KeyError:
        pass
        # print("caught error")
    try:
        myG.remove_edge("a","q")
    except KeyError:
        print("Error appropriately thrown for case when node not in graph")

    myG.remove_node(7)
    print(myG)
    myG.traverse("a")

def traverse_test():
    """ test the traverse function in graph class """
    myG = Graph()
    try:
        myG.traverse("a")
    except KeyError:
        print("appropriate error thrown - Case 1 - myG graph")

    insert = [1,2,3,4,5,6,7,8,9,10]
    for _ in insert:
        myG.add_node(_)
    myG.add_edge(1,2)
    myG.add_edge(1,3)
    myG.add_edge(1,4)
    myG.add_edge(1,6)
    myG.add_edge(2,5)
    myG.add_edge(3,7)
    myG.add_edge(3,8)
    myG.add_edge(4,9)
    myG.add_edge(9,10)
    print(myG)

    # answer should be [1,2,3,4,6,5,7,8,9,10]
    #it's in order except the 5,6 are switched
    if myG.traverse(1) == [1,2,3,4,6,5,7,8,9,10]:
        print("Case 1: successful \n")

    gTwo = Graph() #case #2
    insert = [1,2,3,4,5,6,7,8,9,10]
    for _ in insert:
        gTwo.add_node(_)

    #create edges with for loop and it worked really well
    edges = [1,2,1,4,4,2,1,3,1,5,5,6,5,7,4,8,8,9,9,10]
    for i in range(0,len(edges),2):
        a,b = edges[i], edges[i+1]
        gTwo.add_edge(a,b)

    print(gTwo)
    print(gTwo.traverse(1))
    if gTwo.traverse(1) == [1,2,3,4,5,8,6,7,9,10]:
        print("successful case 2")
    else:
        print("failure with case 2")

    """
    myG.add_edge(1,2)
    myG.add_edge(1,4)
    myG.add_edge(4,2)
    #first three edges make a cycle
    myG.add_edge(1,3)
    myG.add_edge(1,5)
    """
    # myG.add_edge()

def search_shortest_path_Test():
    """ test the shortest path function of graph"""

    threeGraph = Graph() #case #2
    try:
        threeGraph.shortest_path("a",7)
    except KeyError:
        print("appropriate error thrown  when neither node isnt in graph- Case 3 - threeGraph graph \n")

    # test case idea website :: https://www.hackerearth.com/practice/notes/graph-theory-breadth-first-search/
    insert = [1,2,3,4,5,6,7,8,9,10,11]
    for _ in insert:
        threeGraph.add_node(_)

    try:
        threeGraph.shortest_path("a",7)
    except KeyError:
        print("appropriate error thrown  when one node isnt  in graph- Case 3 - threeGraph graph \n")


    #created edges with for loop and it worked really well
    edges = [1,3,3,2,5,2,3,12,12,5,2,4,5,8,4,9,7,9,8,7,8,6,9,10,6,10,10,11]
    if len(edges) % 2 != 0:
        raise ValueError("edges wrong size")
    for i in range(0,len(edges),2):
        a,b = edges[i], edges[i+1]
        threeGraph.add_edge(a,b)
    # print(threeGraph)

    #shortest path from 11 to 1 should be 10:9:4:2:3:1
    answer = threeGraph.shortest_path(11,1)
    # print("Case Three:",answer,sep='\n')
    if answer == [11,10,9,4,2,3,1]:
        print("successful on case 3\n")
    else:
        print("failure on case 3")

    ##########################  next test case ###################################

    # test case idea website - SAME as above - ::
    # https://www.hackerearth.com/practice/notes/graph-theory-breadth-first-search/

    fourGraph = Graph()
    insert = [1,2,3,4,5,6,7,8,9,10]
    for _ in insert:
        fourGraph.add_node(_)
    #create edges with for loop and it worked really well
    # edges = [1/6,1/2,1/8,8/3,6/7,7/9,3/9,3/4,4/5,7/5,9/5] #EASIEST TO CREATE LIKE THIS
    edges = [1,6,1,2,1,8,2,3,8,3,6,7,7,9,3,9,3,4,4,5,7,5,9,5]
    # difference is that 8 doesn't connect to 3
    # edges = [1,6,1,2,1,8,2,3,6,7,7,9,3,9,3,4,4,5,7,5,9,5]
    if len(edges) % 2 != 0:
        raise ValueError("edges wrong size")
    for i in range(0,len(edges),2):
        a,b = edges[i], edges[i+1]
        fourGraph.add_edge(a,b)
    # print(fourGraph)
    #shortest path is 1:6:7:5
    answer = fourGraph.shortest_path(1,5)
    # print("Case FOUR:",answer,sep='\n')
    if answer == [1,6,7,5]:
        print("successful on case 3\n")
    else:
        print("failure on case 4")

class MovieGraph:
    """Class for solving the Kevin Bacon problem with movie data from IMDb."""

    # Problem 4
    def __init__(self, filename="movie_data.txt",):
        """
            Initialize a set for movie titles, a set for actor names, and an
            empty NetworkX Graph, and store them as attributes. Read the speficied
            file line by line, adding the title to the set of movies and the cast
            members to the set of actors. Add an edge to the graph between the
            movie and each cast member.

            Each line of the file represents one movie: the title is listed first,
            then the cast members, with entries separated by a '/' character.
            For example, the line for 'The Dark Knight (2008)' starts with

            The Dark Knight (2008)/Christian Bale/Heath Ledger/Aaron Eckhart/...

            Any '/' characters in movie titles have been replaced with the
            vertical pipe character | (for example, Frost|Nixon (2008)).
        """

        myNet = nx.Graph()
        movies = set()
        actors = set()
        #read in the content
        with open(filename, 'r',encoding = "utf8") as myfile:
            contents = myfile.readlines()
        #segment each line into sublists
        for i in range(len(contents)):
            contents[i] = contents[i].strip().split('/')
            title = contents[i][0]
            list_Actors = contents[i][1:]
            numActors = len(list_Actors)
            movies.add(title)
            myNet.add_node(title)

            #make an edge of each actor in the list of actors to the movie
            for j in range(numActors):
                actor = list_Actors[j]
                myNet.add_node(actor)
                myNet.add_edge(actor,title)
                actors.add(actor)

        self.Network = myNet
        self.movies = movies
        self.actors = actors

    def path_to_actor(self, source, target):
        """
            Compute the shortest path from source to target and the degrees of
            separation between source and target.

            Returns:
                (list): a shortest path from source to target, including endpoints.
                (int): the number of steps from soumyNetrce to target, excluding movies.
        """
        #this returns a list of the shortest path
        minimum_path = list(nx.shortest_path(self.Network, source, target))
        #daniel radcliff, tom hanks is two
        return minimum_path, (len(minimum_path) // 2)

    def average_number(self, target):
        """
            Calculate the shortest path lengths of every actor to the target
            (not including movies). Plot the distribution of path lengths and
            return the average path length.

            Returns:
                (float): the average path length from actor to target.
        """
        average = 0
        sum = 0

        #for each actor calculate the number of movies inbetween them and the target
        # start = time.time()
        dict = nx.shortest_path_length(self.Network,target)
        #exclude the movies from the sum for the average
        people_Dictionary = set(dict).intersection(self.actors)
        values = []
        for i in people_Dictionary:
            distance = dict.get(i) // 2
            sum += distance
            values.append(distance)


        average = sum / len(self.actors)
        # values = dict.values(people_Dictionary)
        plt.hist(values,bins = [i-0.5 for i in range(8)])
        plt.title("Average Number for " + target)
        #or I could use a formatted string f"Average Number for {target}"
        plt.xlabel("Number of Connections")
        plt.ylabel("Number of Actors")
        plt.show()

        # return average, time.time() - start
        return average

def construct_Movie_Graph():
    """ Construct the Movie graph from the data and test it"""

    movies = MovieGraph()
    # print(len(movies.movies))
    # print(len(movies.actors))

    path,number = movies.path_to_actor("Samuel L. Jackson","Kevin Bacon")
    # print(path,number)
    if number == 2:
        print("success with sam jackson and kevin bacon")
    else:
        print("failure with sam jackson and kevin bacon")
    path,number = movies.path_to_actor("Daniel Radcliffe","Tom Hanks")
    # print(path,number)
    if  number == 2:
        print("success with Daniel Radcliffe,Tom Hanks")
    else:
        print("failure with Daniel Radcliffe,Tom Hanks")
    print(movies.path_to_actor("Denzel Washington","Brad Pitt"))

def Mean_test():
    """ test the average number """
    movies = MovieGraph()
    print(movies.average_number("Al Pacino"))

if __name__ == "__main__":
    pass
    # test_graph()
    # traverse_test()
    #search_shortest_path_Test()
    # construct_Movie_Graph()
    # Mean_test()
