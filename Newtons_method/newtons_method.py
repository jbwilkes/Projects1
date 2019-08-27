import numpy as np #used in problem 3
from matplotlib import pyplot as plt
import math #used in test cases of test_newton(), and basins
from autograd import grad #used in test cases for test_newton()

def newton(f, x0, Df, tol=1e-5, maxiter=15, alpha=1.):
    """Use Newton's method to approximate a zero of the function f.

        Parameters:
            f (function): a function from R^n to R^n (assume n=1 until Problem 5).
            x0 (float or ndarray): The initial guess for the zero of f.
            Df (function): The derivative of f, a function from R^n to R^(nxn).
            tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
            maxiter (int): The maximum number of iterations to compute.
            alpha (float): Backtracking scalar (Problem 3).

        Returns:
            (float or ndarray): The approximation for a zero of f.
            (bool): Whether or not Newton's method converged.
            (int): The number of iterations computed.
        """
    prev = x0
    converged = False
    # NOTE: if x0 is ndarray of (1,) shape then np.isscalar returns false - edge case thats not valid for test driver
    #this function will be used in both 1 dimension, and higher dimension computations
    def Zero_Denominator(prev):
        """ flag a ZeroDivisionError in case that Df(prev) = 0,
            although the computations (alpha * f(prev) / Df(prev) ) would probably throw an error
            its good practice to consider, and code up

            Parameters:
                prev (float or ndarray) : previous value in sequence of newtons method calculations

            """
        message = "DF of {} is zero which results division by zero in newtons method".format(prev)
        if np.isscalar(prev):
            if Df(prev) == 0:
                raise ZeroDivisionError(message)
        else:
            if np.allclose(Df(prev), np.zeros_like(prev) ):
                raise ZeroDivisionError(message)
    #I wrote the 1 dimensional version so I just kept it when scaling to higher dimensions instead of having 1 for loop with cases combined

    #1 dimensional case will execute when if statement is True
    if np.isscalar(prev):
        # print(prev)
        for i in range(maxiter):
            # Zero_Denominator(prev)
            new = prev - (alpha * f(prev) / Df(prev) )
            # print(next)
            if abs(prev - new) < tol:
                converged = True
                break
            prev = new
        # return i + 1 instead of i, because i starts at zero, and it's a counter
        return new, converged, i + 1
    #higher dimensional version of newtons method
    else:
        #Yk is the solution to Df(x0)Yk = f(x0) so that new = prev - alpha * Yk
        #in other words next value in sequence results from prev - alpha * YK
        for i in range(maxiter):
            # Zero_Denominator(prev)
            Yk = np.linalg.solve(Df(prev),f(prev))
            #DFx=b is same as x=DF^-1 * b where x is Yk and b is f(x)
            new = prev - (alpha * Yk)
            #the 2 in np.linalg.norm represents the euclidean norm (standard distance)
            if np.linalg.norm(new - prev,2) < tol:
                converged = True
                break
            prev = new
        return new, converged, i + 1

def test_newton():
    """ test the newton function """
    tol = 1e-8 #it could go smaller, but it takes too long
    # tol = 1e-6
    f = lambda x: x**4 - 3
    Df = lambda x: 4*x**3
    initial = 2
    root, convergence, i = newton(f,initial,Df) #roots are  positive and negative: 3^(1/4) or 1.316
    # print("For x^4 - 3 based on intial guess of",initial,"gives approximate root of:",root," which f(root) = ",f(root),"in",i,"iterations  and converged?",convergence)
    assert abs(root - 3**(1/4)) < tol, "test case failed for x**4 - 3"

    #test case #2
    g = lambda x: math.exp(x) - 2
    Dg = lambda x: math.exp(x)
    initial = 1
    root, convergence, i  = newton(g,initial,Dg) #ln(2) ~ 0.693 ??
    # print("For e**x -2+ based on intial guess of",initial,"gives approximate root of:",root," which f(root) = ",g(root),"in",i,"iterations  and converged?",convergence)
    assert abs(root - math.log(2)) < tol, "test case failed for e**x - 2"

    #test backtracking
    #defining cubed root function this way allows for negative x without failures/ errors being thrown
    h = lambda x: np.sign(x) * np.power(np.abs(x), 1./3) #cubed root of x
    Dh = grad(h)
    initial = 0.01
    alpha = 1.
    #it shouldn't converge for alpha = 1
    root, convergence, i = newton(h,initial,Dh,1e-5,15,alpha)
    assert (convergence is False), "for alpha = 1 cubed root converged when it shouldn't"

    initial = 0.01
    alpha = .4
    #it should converge for alpha = 1
    root, convergence, i = newton(h,initial,Dh,1e-5,15,alpha)
    # print("For cubed root function based on intial guess of",initial,"gives approximate root of:",root," which h(root) = ",h(root),"in",i,"iterations  and converged?",convergence)
    real_root = 0
    #it's true that this test case (the 9e-3) was constructed from the output, but because
    #I can reason that its accurate for an approximation, this test case is just in case
    #I make adjustments and want to verify ouput is still satisfactory
    # print("For cubed root based on intial guess of",initial,"gives approximate root of:",root," which h(root) = ",h(root),"in",i,"iterations  and converged?",convergence)
    assert abs(root - real_root) < 9e-3, "failed for alpha = 0.4 with cubed root function"

    #test higher dimensional functions
    #this test case comes from lecture notes (day 09) in math 346
    #f maps 2D vectors to 2D vectors
    #having a period on some of the elements make them float ndarrays
    f = lambda x: np.array([x[0]**2 - x[1],x[0] + x[1] - 1.])
    #construct the derivative matrix so that it can accept 2d vectors
    Df = lambda x: np.array([[2*x[0], -1. * x[1]**0],[1., 1.]])
    initial = np.array([.5, .5])
    # output should be the following vector [0.61 , 0.38]
    #math.sqrt evaluates faster than **(1/2)
    real_root = np.array([(-1 + math.sqrt(5)) / 2 , (3 - math.sqrt(5)) / 2])
    root, convergence, i = newton(f,initial,Df)
    # print("For 2D function based on intial guess of",initial,"gives approximate root of:",root," which f(root) = ",f(root),"in",i,"iterations  and converged?",convergence)
    assert np.linalg.norm(root - real_root) < tol, "2 dimensional case failed "

    #test a divide by zero (critical point - or maximum of a function)
    f = lambda x: -3 * x**2 + 57 * x
    Df = lambda x: -6*x + 57
    initial = 9.5 #this is the local maximum and should throw an error
    try:
        root, convergence, i = newton(f,initial, Df)
    except ZeroDivisionError:
        pass
        # print("division by zero test case worked")

    #this print statement will evaluate only if there are no "unexpected" errors
    print("no errors for test cases of newton() function")

def prob2(N1, N2, P1, P2):
    """Use Newton's method to solve for the constant r that satisfies
        P1[(1+r)**N1 - 1] = P2[1 - (1+r)**(-N2)].
        With r_0 = 0.1 for the initial guess.

        Parameters:
            P1 (float): Amount of money deposited into account at the beginning of
            years 1, 2, ..., N1.
            P2 (float): Amount of money withdrawn at the beginning of years N1+1,
            N1+2, ..., N1+N2.
            N1 (int): Number of years money is deposited.
            N2 (int): Number of years money is withdrawn.

        Returns:
            (float): the value of r that satisfies the equation.
        """
    r0 = 0.1
    #find the root of the equation f solving for r,
    f = lambda r: P1 * (-1 + (1+r)**(N1)) - P2 * (1 - (1+r)**(-1*N2))
    Df = lambda r:  P1 * N1 * (1+r)**(N1 - 1) - P2 * N2 * (1+r)**(-1*N2 - 1)
    #increase number of possible iterations to allow for convergence
    root, convergence, iterations = newton(f,r0,Df,maxiter=30)
    if convergence is True:
        return root
    else:
        print("if it doesn't converge then should I raise an error")
        return root

def optimal_alpha(f, x0, Df, tol=1e-5, maxiter=15):
    """Run Newton's method for various values of alpha in (0,1].
        Plot the alpha value against the number of iterations until convergence.

        Parameters:
            f (function): a function from R^n to R^n (assume n=1 until Problem 5).
            x0 (float or ndarray): The initial guess for the zero of f.
            Df (function): The derivative of f, a function from R^n to R^(nxn).
            tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            (float): a value for alpha that results in the lowest number of
            iterations.
        """
    num_alphas = 10000
    #do not include zero in interval,
    #use linspace(1,0,num_alphas,endpoint=False) that will simulate interval (0,1]
    alphas = np.linspace(1,0,num_alphas,endpoint=False)
    iteration_count = maxiter * np.ones_like(alphas)
    minimum_iterations = maxiter
    minimum_alpha = 1
    #find minimum iterations and minimum alpha
    for i in range(num_alphas):
        tester = alphas[i]
        root, convergence, iterations = newton(f,x0,Df,tol,maxiter,tester)
        iteration_count[i] = iterations
        if iterations < minimum_iterations:
            minimum_iterations = iterations
            minimum_alpha = tester
    #plot alphas to iterations
    plt.plot(alphas, iteration_count, '-b')
    plt.xlabel("Alpha Values")
    plt.ylabel("Iteration Result")
    plt.axis((0,1,0,maxiter))
    plt.title("Optimal Alpha out of {} Trials".format(num_alphas))
    plt.show()

    return minimum_alpha

def test_optimal_alpha():
    """ test the optimal alpha function """
    h = lambda x: np.sign(x) * np.power(np.abs(x), 1./3) #cubed root of x
    Dh = lambda x: 1 / (3 * np.power(np.abs(x),2./3) )
    guess = 1
    #output should near .3 rather than .4 for cubed root function, is really close to .2
    cubed_root_output = optimal_alpha(h,guess, Dh)
    print("\nOptimal Alpha Output for cubed root function is {}".format(cubed_root_output))
    assert abs(cubed_root_output - 0.33429999999999993) < 1e-7, "test case failed for cubed root function"
    print("passed cubed root optimum alpha test")

def prob6():
    """Consider the following Bioremediation system:
                              5xy − x(1 + y) = 0
                        −xy + (1 − y)(1 + y) = 0
        Find an initial point such that Newton’s method converges to either
        (0,1) or (0,−1) with alpha = 1, and to (3.75, .25) with alpha = 0.55.
        Return the intial point as a 1-D NumPy array with 2 entries.
        """
    f = lambda x : np.array([5.*x[0]*x[1] - x[0]*(1 + x[1]), -1*x[0]*x[1] + (1. - x[1])*(1 + x[1])])
    Df = lambda x: np.array([[4.*x[1] - 1 , 4*x[0]],[-1*x[1], -1 * x[0] - 2*x[1]]])
    iteration_tol = 15 #i.e. maxiter
    tol = 1e-5
    root1 = np.array([0.,1])
    root2 = np.array([0.,-1])
    root3 = np.array([3.75,0.25])
    first_alpha = 1. #yes it's default of newton function but I want readibility & consistency
    second_alpha = 0.55
    num_alphas = 200
    # alpha_one_convergence = False
    # alpha_two_convergence = False
    #i'll first shift the domain, then scale it using array broadcasting
    # domain_scaler = np.array([1./4,1.])
    # domain_shifter = np.array([-1.,1/4])
    #continue looping until alpha_one and alpha_two converge
    x_val = np.linspace(-1/4,0,num_alphas)
    y_val = np.linspace(0,1/4,num_alphas)
    #systematically search the domain
    for x in x_val:
        for y in y_val:
            test_vector = np.array([x,y])
            """ #Getting a random vector is more likely than systematic checking domain #NOTE: wrong!
                    #two is the dimension of the returning vector i.e. shape will be (2,)
                    # sample = np.random.uniform(0,1,(2,))
                    #using normal vector addition and elementwise multiplication
                    # thus the test vectors are in domain [-.25, 0] x [0,.25]
                    # test_vector = np.multiply((sample + domain_shifter), domain_scaler)
                    """
            #perform newtons method on this test vector
            #see if test vector converges within iteration tolerance
            #if it converges for one alpha then test the second alpha
            #[0] is an extractor of the tuple (root, convergence boolean, and iteration)
            root_test = newton(f,test_vector,Df,maxiter=iteration_tol,alpha=first_alpha)[0]
            if (np.allclose(root_test, root1) is True ) or (np.allclose(root_test, root2) is True):
            # if (np.linalg.norm(root_test - root1) < tol ) or (np.linalg.norm(root_test - root2) < tol ):
                root_test2 = newton(f,test_vector,Df,maxiter=iteration_tol,alpha=second_alpha)[0]
                if (np.allclose(root_test2, root3) is True):
                # if np.linalg.norm(root_test2 - root3) < tol:
                    return test_vector
                    break
    return "failed to converge"

def plot_basins(f, Df, zeros, domain, res=1000, iters=15):
    """Plot the basins of attraction of f on the complex plane.
        Parameters:
            f (function): A function from C to C.
            Df (function): The derivative of f, a function from C to C.
            zeros (ndarray): A 1-D array of the zeros of f.
            domain ([r_min, r_max, i_min, i_max]): A list of scalars that define
            the window limits and grid domain for the plot.
            res (int): A scalar that determines the resolution of the plot.
            The visualized grid has shape (res, res).
            iters (int): The exact number of times to iterate Newton's method.
        """
    # x_real = np.linspace(-1.5,1.5,1000)
    # x_imag = np.linspace(-1.5,1.5,1000)
    x_real = np.linspace(domain[0],domain[1],res)
    x_imag = np.linspace(domain[2],domain[3],res)
    X_real , X_imag = np.meshgrid(x_real,x_imag)
    X_0 = X_real + 1j*X_imag
    prev = X_0
    #perform newtons method
    #for efficienty don't use newton() function
    for i in range(iters):
        new = prev - f(prev) / Df(prev)
        prev = new
    X_k = new
    n,m = np.shape(X_k)
    #Y is going to denote color in the graph corresponding to a specific zero
    Y = np.ones_like(X_k)
    #for each (i)th row
    for i in range(n):
        #for each (j)th column
        for j in range(m):
            #array broadcast subtract array of zeros by scalar X_k[i][j]
            Y[i][j]  = np.argmin(np.abs(zeros - X_k[i][j]))
    #plot the basins
    plt.pcolormesh(X_real, X_imag,np.real(Y),cmap="brg")
    # plt.pcolormesh(X_real, X_imag,Y,cmap="rainbow")
    plt.title("Root Basins of Attraction")
    plt.show()

def test_cases():
    """ test cases for various functions """
    test_newton() #it passes all my test cases
    # print(abs(prob2(30,20,2000,8000) - 0.03878)) #should be small when correct
    assert abs(prob2(30,20,2000,8000) - 0.03878) < 1.7e-06, "prob2 test case failed for (30,20,2000,8000) as input"
    print("passed problem 2 test case ")
    test_optimal_alpha() #has test cases in it
    print("finished \"test cases\" function output ")
    print(prob6())

def basins_test():
    """ basins function test """
    f = lambda x: x**3 - 1
    Df = lambda x: 3*x**2
    zeros_f = np.array([1.,
                        (-1/2) + 1j * math.sqrt(3) / 2,
                        (-1/2) - 1j * math.sqrt(3) / 2])
    a = 1.5
    domain_f = [-a,a,-a,a]
    plot_basins(f,Df,zeros_f,domain_f)

    g = lambda x: x**3 - x
    Dg = lambda x: 3*x**2 - 1
    zeros_g = np.array([-1.,0,1])
    a = 1.5
    domain_g = [-a,a,-a,a]
    plot_basins(g,Dg,zeros_g,domain_g)

if __name__ == "__main__":
    pass
    test_optimal_alpha()
    # test_cases()
    # basins_test()
