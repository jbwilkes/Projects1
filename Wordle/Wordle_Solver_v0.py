import pandas as pd, numpy as np, os,glob
import string
from matplotlib import pyplot as plt
import seaborn as sns # for heatmap

class Wordle_Solver():
    """ """
    def __init__(self,wordle_length=5
                 ,dataframe_of_words=None
                 ,set_of_words_path=None
                 
                 ):
        """ """
        print('update all the docstrings')
        print('test display remaining words')
        print('build in a check to see if the guess_results has a contradiction? it was entered incorrectly?')
        print('\n')
        # self.guess_history = """ """
        self.wordle_length = wordle_length
        self.guess_history = []
        self.in_word = set() ## maintain as a set to avoid duplicates
        self.not_in_word = set() ## maintain as a set to avoid duplicates
        #info_per_letter_slot = {'green':None,'yellow':[]}
        ## for each slot in the word length, have if it's green then letter - else list, or if its yellow then add to list
        self.letter_slots = {i:{'green':'','yellow':[]} for i in range(1, wordle_length + 1)} ##arbitrary choice to have 1 based index as opposed to zero
        self.remaining_viable_letters = list(string.ascii_lowercase)

        if dataframe_of_words is not None:
            ## there is an assumed structure of this dataset that will be described elsewhere
            self.df = dataframe_of_words
        else:
            if set_of_words_path is None:
                set_of_words_path = f'{wordle_length} letter words parsed - NLTK Words Corpora - v0.csv'

                if set_of_words_path not in os.listdir():
                    raise NotImplementedError('Download the dataset')

            ## there is an assumed structure of this dataset that will be described elsewhere
            df = pd.read_csv(set_of_words_path)
            ## recall dataframes are mutable 
            self.df = df

        ## remaining indices wont be helpful actually, because 
        ## the solver needs the intersection of indices of all conditions, including from past guesses
        ## in order to narrow down to the correct word. 
        self.remaining_indices = set(self.df.index)

        ##
        self.total_number_of_guesses = 0

        ## the dataset currently uses strings denoting the cardinal placement as column name for letter slot of the word
        ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
        self.cardinals = {n:ordinal(n) for n in range(1,wordle_length + 1)}

    def print_remaining_viable_letters(self):
        """ """
        pass 
        return self.remaining_viable_letters
    
    def display_remaining_words(self):#,remaining_words):
        """ """
        # pass
        for i,x in enumerate(self.remaining_words):
            i += 1
            if i % 5 == 0:
                print(x)
            else:
                print(x,end=', ')

    def new_guess_results(self,guess_results
                  ,return_remaining_words=True
                  ,display_remaining_words=False
                  ,display_remaining_letter_frequency=True):
        """ """
        self.total_number_of_guesses += 1
        # self.guess_history.append(guess_results)
        ## stopping criteria, if the word has been identified, still add final guess to history 

        ## maybe create a history of the guess (without the results), and the results - just so it's easy to read 
        
        #update remaining viable letters, only remove the blacks.

        # parse the guess (with results) to get the guess (without results) and store it, for maximum simplicity - the user doesn't want to enter in the input with results & without results. 
        ## then an assertion can be added in that the length of the guess has to be equal to the wordle_length, its just to make sure the guess_results is typed correctly

        ## parse through the results
        ## add in a '*' to the end of the guess results to avoid Error
        guess_results += '*'
        letter_slots_counter = min(self.letter_slots.keys()) #1 # the letter_slot counter arbitrarly start at 1
        number_of_greens = 0
        for i,element in enumerate(guess_results[:-1]):
            if element not in {'!','?','*'}:
                ## the dataset has lowercase 
                element = element.lower()
                ## see what the result is 
                if guess_results[i + 1] == '!':
                    #it's green 
                    self.in_word.add(element)
                    ## add 
                    self.letter_slots[letter_slots_counter]['green'] = element
                    number_of_greens += 1
                elif guess_results[i + 1] == '?':
                    #if yellow, then it is in the word
                    self.in_word.add(element)
                    ## this slot is not this letter
                    self.letter_slots[letter_slots_counter]['yellow'].append(element)
                else:
                    # if not green or yellow, then its black
                    self.not_in_word.add(element)
                    ## we could add to letter_slots but its not needed for black because adding to not_in_word has same affect
                    ##update remaining letters 
                    try:
                        self.remaining_viable_letters.remove(element) 
                    except:
                        pass
                        ## may have already been removed
                letter_slots_counter += 1

        ## have to ensure that duplicate letters that are marked black dont negate the fact that there is the first instance in the word 
        ## like first word guess "wea!r!y" and second guess "apa!r!t" - with wordle 'shard', not that in the second guess the first letter 'a' is black. Dont want it in the not_in_word list
        self.not_in_word -= self.in_word

        ## stopping criteria
        if number_of_greens == self.wordle_length:
            #print('\n\nyou solved it with guess',guess_results[:-1])
            print('\n\nyou solved it')
            
        #     print('\n\n',self.letter_slots)
        #     print('\n\n',self.in_word)
        #     print('\n\n',self.not_in_word)
            return [''.join([x for x in guess_results[:-1] if x != '!'])]

        #for simplicity
        df = self.df

        ## UPDATE THE DATASET TO BE HAVE NUMBER OF TIMES THE LETTER SHOWS UP, NOT JUST A BOOLEAN AS TO WHETHER THE LETTER IS IN THE WORD
        for iw in self.in_word:
            self.remaining_indices = self.remaining_indices.intersection(set(df.loc[df[iw.lower()] == 1].index))
        for niw in self.not_in_word:
            self.remaining_indices = self.remaining_indices.intersection(set(df.loc[df[niw] == 0].index))
        
        ## now update the letter slots 
        ##cls for cardinal letter slot - which is the column name
        ##ols for ordinal letter slot 
        
        # print(self.letter_slots)
        # print(self.not_in_word)

        for ols in self.cardinals.keys():
            letter_slots_counter = ols 
            cls = self.cardinals[ols]
            
            if self.letter_slots[ols]['green'] != '':
                # then we know what letter is green in this slot 
                #print('green at',cls,'and is ',self.letter_slots[letter_slots_counter]['green'])
                self.remaining_indices = self.remaining_indices.intersection(set(df.loc[df[cls] == self.letter_slots[letter_slots_counter]['green']].index))
            elif self.letter_slots[ols]['yellow'] != []:
                #print('yellow at',cls,'and is ',list(set(self.letter_slots[letter_slots_counter]['yellow'])))
                ## there are yellow letters here, denoting it's not this letter
                self.remaining_indices = self.remaining_indices.intersection(set(df.loc[~df[cls].isin(set(self.letter_slots[letter_slots_counter]['yellow']))].index))

        remaining_words = list(df.iloc[list(self.remaining_indices)]['word'].values)
        self.remaining_words = remaining_words

        print('number of remaining words is:',len(remaining_words))
        if display_remaining_letter_frequency:
            self.remaining_letter_frequency()
        if display_remaining_words:
            self.display_remaining_words()#remaining_words)
        if return_remaining_words:
            return remaining_words

    def remaining_letter_frequency(self):
        """ """
        #in_word = list('rne') ## top_left
        # in_word = list('tha') ## top_right
        # in_word = list('o') ## bottom_left
        # in_word = list('tus') ## bottom_right

        rwdf = pd.DataFrame([list(x) for x in self.remaining_words])
        # rwdf = pd.DataFrame([list(x) for x in top_right])
        # rwdf = pd.DataFrame([list(x) for x in bottom_left])
        # rwdf = pd.DataFrame([list(x) for x in bottom_right])

        letters, counts = np.unique(rwdf.values.ravel(),return_counts=True)
        # plt.bar(letters,counts)
        # plt.show()
        S = pd.Series(counts,index=letters)
        S.drop(self.in_word,inplace=True)
        S.sort_index().plot(kind='barh')
        plt.show()
        pass

    def random_solver(self):
        """ """
        # instantiate the Wordle Game, have it pick a random word as the WORDLE
        # pick a random word to guess with, 

        #return history of guesses, how many guesses did it take total
        pass

    
def test_class():
    """ """
    wordle = 'cactus'
    guesses_results = {'canyon':None
                       ,'claims':None
                       ,'cactus':None
                       }

    # wordle = 'inert'
    guesses_results = {'yarns':'yar?n?s'
                       ,'honor':'hon?or?'
                       ,'nerve':'n?e?r?ve'
                       ,'inure':'i!n!ur!e?'
                       ,'inert':'i!n!e!r!t!'
                       }

    guesses_results = {'candy':'ca?ndy'
                       ,'after':'a?fter!'
                       ,'solar':'s!ola!r!'
                       ,'sugar':'s!u!g!a!r!'
                       }
    ## january 27th
    ## this gets it down to one remaining word
    guesses_results = {'learn':'learn?'
                       ,'sound':'so!u!n!d'
                       ,'ticks':'t?icks'
                       ,'mount':'m!o!u!n!t!'
                    #    ,'':
                       }
    
    ##shard helped me remember to remove some values from not_in_word variable
    ## this zeros out after the first 2 
    ## february 3rd
    ## after the second guess there should be 34 words left
    guesses_results = {'weary':'wea!r!y'
                       ,'apart':'apa!r!t'
                       ,'board':'boa!r!d!'
                       ,'guard':'gua!r!d!'
                       ,'shard':'s!h!a!r!d!'
                       #,'':''
                       }

    ## template
    # guesses_results = {'':''
    #                    ,'':''
    #                    ,'':''
    #                    ,'':''
    #                    ,'':''
    #                    ,'':''
    #                    }
    
    ## I have tons of test cases from taking pictures from playing wordle 
    WS = Wordle_Solver(wordle_length=5
                 ,set_of_words_path=None)
    
    for gr in guesses_results.values():
        remaining_words = WS.new_guess_results(guess_results=gr
                  ,return_remaining_words=True
                  ,display_remaining_words=False)
        if len(remaining_words) < 10:
            print(remaining_words)
        # WS.new_guess_results(guess_results=gr
        #           ,return_remaining_words=False
        #           ,display_remaining_words=False)
        
# test_class()
