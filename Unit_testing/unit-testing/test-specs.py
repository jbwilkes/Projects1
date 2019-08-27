import specs
import pytest


def test_add():
    assert specs.add(1, 3) == 4, "failed on positive integers"
    assert specs.add(-5, -7) == -12, "failed on negative integers"
    assert specs.add(-6, 14) == 8

def test_divide():
    assert specs.divide(4,2) == 2, "integer division"
    assert specs.divide(5,4) == 1.25, "float division"
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.divide(4, 0)
    assert excinfo.value.args[0] == "second input cannot be zero"

# Problem 1: write a unit test for specs.smallest_factor(), then correct it.
def test_smallest_factor():
    """
    cases:
        n = 15, return 3                            a composite number
        n = 23, return 23 (since it's prime)        a prime numbers, square root of 23 is less than 5.
        n = 7, return 7                             square root of 7 is 3
        n = 16, return 2 (NOT 4), since 2|16 = 8.
        n = 221, return 13 (since 13*17 is 221)
    """
    assert specs.smallest_factor(23) == 23, "failed on prime number case"
    assert specs.smallest_factor(16) == 2, "failed third case"
    assert specs.smallest_factor(7) == 7, "failed last case"
    assert specs.smallest_factor(221) == 13, "failed composite of prime numbers case"
    assert specs.smallest_factor(15) == 3, "failed composite number case"
    #print("it got to this point")

# Problem 2: write a unit test for specs.month_length().
def test_month_length():
    #these loops will test all the months
    for _ in ["September", "April", "June", "November"]:
        assert specs.month_length(_) == 30, "30 day month error"
    for _ in ["January", "March", "May", "July","August", "October", "December"]:
        assert specs.month_length(_) == 31, "31 day month error"
    assert specs.month_length("February", False) == 28, "feb, non leap year error"
    assert specs.month_length("February", True) == 29, "leap year error, feb"
    #make sure to test the case when an unexpected string is passed through the function
    assert specs.month_length("not month string") is None, "Not month tested"

# Problem 3: write a unit test for specs.operate().
def test_operate():
    assert specs.operate(3,4,"+") == 7, "addition operation failed"
    assert specs.operate(5,4,"-") == 1, "subtraction operation failed"
    assert specs.operate(3,4,"*") == 12, "multiplication operation failed"
    assert specs.operate(12,3,"/") == 4, "Division failed"
    pytest.raises(TypeError, specs.operate, a=3,b=3,oper=3)
    pytest.raises(ValueError, specs.operate, a=3,b=3,oper="^")
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.operate(4,0,"/")
    #FOR SOME REASON THE MESSAGE THAT IS RETURNED with the error in the assert must match
    #the error that is given by the raise in spec.py - INTERESTING. My hypothesis was correct.
    assert excinfo.value.args[0] == "division by zero is undefined"

# Problem 4: write unit tests for specs.Fraction, then correct it.
@pytest.fixture
def set_up_fractions():
    """
    the add is weird
    self(1,5) is 1/5, and other is 2/3
    to add them take 3/3*(1/5) and 5/5*(2/3)
    numerator will become (3*1 + 5*2), denominator is 3*5.

    """

    frac_1_3 = specs.Fraction(1, 3)
    frac_1_2 = specs.Fraction(1, 2)
    frac_n2_3 = specs.Fraction(-2, 3)
    return frac_1_3, frac_1_2, frac_n2_3

def test_fraction_init(set_up_fractions):
    #this function will also test the Errors
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_3.numer == 1
    assert frac_1_2.denom == 2
    assert frac_n2_3.numer == -2
    frac = specs.Fraction(30, 42)
    pytest.raises(ZeroDivisionError, specs.Fraction,  numerator = 7
    , denominator = 0)
    assert frac.numer == 5
    assert frac.denom == 7
    with pytest.raises(TypeError) as excinfo:
        specs.Fraction(7, "hello")
    assert excinfo.value.args[0] == "numerator and denominator must be integers"

def test_fraction_str(set_up_fractions):
    #this includes the case when denominator is 1
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    frac = specs.Fraction(3,1)
    assert str(frac) == "3"
    assert str(frac_1_3) == "1/3"
    assert str(frac_1_2) == "1/2"
    assert str(frac_n2_3) == "-2/3"

def test_fraction_float(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert float(frac_1_3) == 1 / 3.
    assert float(frac_1_2) == .5
    assert float(frac_n2_3) == -2 / 3.

def test_fraction_eq(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 == specs.Fraction(1, 2)
    assert specs.Fraction.__eq__(frac_1_2, 0.5)
    assert frac_1_3 == specs.Fraction(2, 6)
    assert frac_n2_3 == specs.Fraction(8, -12)

def test_fraction_add(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert str(specs.Fraction.__add__(frac_1_3,frac_1_2)) == "5/6"
    assert str(specs.Fraction.__add__(frac_1_3,frac_n2_3)) == "-1/3"

def test_fraction_sub(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert specs.Fraction.__sub__(frac_1_3,frac_n2_3).numer == 1
    assert str(specs.Fraction.__sub__(frac_1_3,frac_n2_3)) == "1"

def test_fraction_mul(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert specs.Fraction.__mul__(frac_1_3,frac_1_2) == 1/6
    assert specs.Fraction.__mul__(frac_1_3,frac_n2_3) == -2/9

def test_fraction_truD(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert str(specs.Fraction.__truediv__(frac_1_3,frac_1_2)) == "2/3"
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.Fraction.__truediv__(frac_1_3,specs.Fraction(0,1))
    assert excinfo.value.args[0] == "cannot divide by zero"

# Problem 5: Write test cases for Set.
def test_count_set():
    #this first and last card are not unique in hand "x"
    x = ["1022", "1122", "0100", "2021",
                "0010", "2201", "2111", "0020",
                "1102", "0210", "2110", "1022"]
    #the last card in this  "y" hand does not have exactly 4 digits
    y = ["1022", "1122", "0100", "2021",
                "0010", "2201", "2111", "0020",
                "1102", "0210", "2110", "10200"]
    #the third card in this "z" hand, is not in base 3
    z = ["1022", "1122", "0800", "2021",
                "0010", "2201", "2111", "0020",
                "1102", "0210", "2110", "1020"]

    #this is the example on page 106,
    #which has 6 sets
    hand1 = ["1022", "1122", "0100", "2021",
                "0010", "2201", "2111", "0020",
                "1102", "0210", "2110", "1020"]

    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(["1001"])
    assert excinfo.value.args[0] == "there are not exactly 12 cards"

    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(x)
    assert excinfo.value.args[0] == "the cards are not all unique"

    string3 = "one or more cards does not have exactly 4 digits"
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(y)
    assert excinfo.value.args[0] == string3

    string4 = "one or more cards has a character other than 0, 1, or 2"
    with pytest.raises(ValueError) as excinfo:
        specs.count_sets(z)
    assert excinfo.value.args[0] == string4

    assert specs.count_sets(hand1) == 6, "incorrect count of sets"


def test_is_set():
    #it should fail on the third digit
    assert specs.is_set("1012","1022","1012") is False, "false negative"
    assert specs.is_set("1022","1121","1220") is True, "unidentified set"

#if __name__ == "__main__":
#    test_smallest_factor()
