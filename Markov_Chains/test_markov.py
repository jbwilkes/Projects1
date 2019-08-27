"""

Markov Chains Test File

"""

import markov_chains as mc
import numpy as np
import pytest

@pytest.fixture
def set_up_4x4():
    """
    The transition matrix

              Hot  Mild Cold Freezing
    Hot      [0.5, 0.3, 0.1, 0  ],
    Mild     [0.3, 0.3, 0.3, 0.3],
    Cold     [0.2, 0.3, 0.4, 0.5],
    Freezing [0,   0.1, 0.2, 0.2]

    """

    four_state = np.array(
        [
            [0.5, 0.3, 0.1, 0  ],
            [0.3, 0.3, 0.3, 0.3],
            [0.2, 0.3, 0.4, 0.5],
            [0,   0.1, 0.2, 0.2]
        ])

    return four_state

@pytest.fixture
def set_up_2x2():

    # test case from book
    weather = np.array([
        [0.7, 0.6],
        [0.3, 0.4]
    ])

    weather_steady_state = np.array([2/3, 1/3])

    return weather, weather_steady_state

def test_random_chain():

    # Check that each column sums to 1

    n = 10
    markov_matrix = mc.random_chain(n)
    for i in range(n):
        assert np.allclose( markov_matrix[:, i].sum(), 1)

def test_forcast(set_up_2x2):
    transition, correct_answer = set_up_2x2

    # Test if all the values are zeroes and ones... how else could you test this???
    n = 10
    weather = mc.forecast(n)

    print(weather)
    assert all([weather[i] in [0, 1] for i in range(n)])

    steady_state_x = mc.steady_state(transition)
    # Make sure steady_state calculated from problem 4 is correct
    assert(np.allclose(transition @ steady_state_x, steady_state_x))

    trials = int(1e4)
    results = mc.forecast(trials)
    average_hot_days = np.array(results).sum() / trials

    assert np.allclose(average_hot_days, correct_answer[1], atol=.01)


def test_four_state_forcast(set_up_4x4):
    # get 4x4 matrix
    weather_matrix = set_up_4x4

    # number of samples
    n = int(1e5)
    weather = mc.four_state_forecast(n)

    # Test: is each entry valid?
    assert all( [weather[i] in [0, 1, 2, 3] for i in range(n)] )


    # Test: does the steady state match
    steady_state = mc.steady_state(weather_matrix)

    # count number of 0's, 1's, 2's, and 3's and divide by length of weather
    answer = np.array([ # list comprehension
        sum( [weather[j] == i for j in range(len(weather))] ) / len(weather)
        for i in [0, 1, 2, 3]
    ])

    print(answer)
    assert np.allclose(steady_state, answer, atol=.005)

def test_steady_state(set_up_2x2):

    weather, correct_answer = set_up_2x2

    # Test case using problem 1 to generate a steady state matrix.
    markov_matrix = mc.random_chain(4)
    steady_state_x = mc.steady_state(markov_matrix)
    assert(np.allclose(markov_matrix @ steady_state_x, steady_state_x))

    # Test weather 2Ã—2 example from book
    assert np.allclose(mc.steady_state(weather), correct_answer)

"""
def test_SentenceGenerator_init():
    generator = mc.SentenceGenerator("yoda.txt")

    print(generator.tranition)

    assert False
"""
