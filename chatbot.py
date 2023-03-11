# PA7, CS124, Stanford
# v.1.0.4
#
# Original Python code by Ignacio Cases (@cases)
######################################################################
import util

import sys, os
import numpy as np
import shlex
import math
import string
import re
#from deps.nltk import nltk
#from deps import nltk
from porter_stemmer import PorterStemmer
import random


# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    edge_cases = {'enjoyed': 'enjoy', 'enjoying': 'enjoy', 'enjoys': 'enjoy', 'loved': 'love', 'loving': 'love', 'loves': 'love', 'hated': 'hate', 'hates': 'hate', 'hating': 'hate', 'disliked': 'dislike', 'dislikes': 'dislike', 'disliking': 'dislike', 'liked': 'like', 'likes': 'like', 'liking': 'like', 'despised': 'despise', 'despises': 'despise', 'despising': 'despise', 'adored': 'adore', 'adores': 'adore', 'adoring': 'adore'}
    negators = ['not', 'didnt', 'never', 'cant', 'couldnt', 'shouldnt', 'dont']
    negators_conj = ['but', 'yet', 'nonetheless', 'although', 'despite', 'however', 'nevertheless', 'still', 'though', 'unless', 'unlike', 'until', 'whereas']
    emphasizers = ['really', 'reaally', 'very', 'extremely', 'totally', 'completely', 'absolutely', 'utterly', 'perfectly', 'entirely', 'thoroughly', 'completely', 'utterly', 'fully', 'wholly', 'altogether', 'entirely', 'fully', 'perfectly', 'quite', 'rather', 'somewhat', 'too', 'utterly', 'very', 'awfully', 'badly', 'completely', 'considerably', 'decidedly', 'deeply', 'enormously', 'entirely', 'especially', 'exceptionally', 'extremely', 'fiercely', 'flipping']
    strong_positive = ['love', 'adore', 'amazing', 'awesome', 'great', 'fantastic', 'wonderful', 'excellent', 'perfect', 'superb', 'terrific', 'brilliant', 'outstanding', 'fabulous', 'marvelous', 'splendid', 'stunning', 'amazing', 'astonishing', 'astounding', 'awesome', 'breathtaking', 'cool', 'dazzling', 'delightful', 'excellent', 'extraordinary', 'fabulous', 'fantastic', 'fine', 'first-class', 'first-rate', 'flawless', 'fortunate', 'fortunate', 'fortunate']
    strong_negative = ['hate', 'despise', 'terrible', 'awful', 'horrible', 'bad', 'lousy', 'poor', 'abysmal', 'atrocious', 'awful', 'bad', 'bogus', 'cheap', 'crappy', 'crummy', 'cruddy', 'crude', 'deplorable', 'dreadful', 'dumb', 'dysfunctional', 'embarrassing', 'enormous', 'evil']

    def __init__(self, creative=False):
        # The chatbot's default name is `moviebot`.
        # TODO: Give your chatbot a new name.
        self.name = 'Bert'

        self.creative = creative

        # This matrix has the following shape: num_movies x num_users
        # The values stored in each row i and column j is the rating for
        # movie i by user j
        self.titles, ratings = util.load_ratings('data/ratings.txt')
        self.sentiment = util.load_sentiment_dictionary('data/sentiment.txt')
        self.movies = util.load_titles('data/movies.txt')
        self.prev_movies = []
        self.recommended_movies = []
        self.recommendations = []
        self.movie_count = 0
        self.num_recs_given = 0
        

        ########################################################################
        # TODO: Binarize the movie ratings matrix.                             #
        ########################################################################

        # Binarize the movie ratings before storing the binarized matrix.
        self.ratings = Chatbot.binarize(ratings)
        # create a 1-d array of user ratings
        self.user_ratings = np.zeros(ratings.shape[0])
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    ############################################################################
    # 1. WARM UP REPL                                                          #
    ############################################################################

    def greeting(self):
        """Return a message that the chatbot uses to greet the user."""
        ########################################################################
        # TODO: Write a short greeting message                                 #
        ########################################################################

        if self.creative == True:
            greeting_message = "Hi! I'm Bert, a movie recommender bot! ♡⸜(˶˃ ᵕ ˂˶)⸝♡ Tell me about a movie, and once Bert understands your taste in movies, Bert will try his best to recommend a movie you'll like! Also, put the title in quotation marks please! It'll make Bert's life easier. (人・ェ・) If Bert misunderstands anything, tell Bert!"
        else:
            greeting_message = "Hey there, I'm Bert, a movies recommender bot. Tell me about a movie, and once I feel like I have a solid of understanding of your taste in movies, I'll try my best to recommend you movies you'd like. If I misunderstand anything, feel free to correct me. Also, please include the title of the movie in quotation marks."
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return greeting_message

    def goodbye(self):
        """
        Return a message that the chatbot uses to bid farewell to the user.
        """
        ########################################################################
        # TODO: Write a short farewell message                                 #
        ########################################################################
        if self.creative == True:
            goodbye_message = "Bye bye! Bert hopes he was helpful. (｡•̀ᴗ-)✧"
        else:
            goodbye_message = "Bye! I hope you enjoy watching whatever movie you end up choosing!"
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################
    def check_quotation_marks(self, s):
        """
        This helper function checks if the string has closed quotation marks.
        Citation: https://stackoverflow.com/questions/64539353/algo-to-check-that-a-string-doesnt-have-missing-close-quote-or
        """
        lst = []
        for char in s:
            if char == '"':
                if lst and lst[-1] == char:
                    lst.pop()
                else:
                    lst.append(char)
            else:
                pass

        return len(lst) == 0

    def process(self, line):
        """Process a line of input from the REPL and generate a response.

        This is the method that is called by the REPL loop directly with user
        input.

        You should delegate most of the work of processing the user's input to
        the helper functions you write later in this class.

        Takes the input string from the REPL and call delegated functions that
          1) extract the relevant information, and
          2) transform the information into a response to the user.

        Example:
          resp = chatbot.process('I loved "The Notebook" so much!!')
          print(resp) // prints 'So you loved "The Notebook", huh?'

        :param line: a user-supplied line of text
        :returns: a string containing the chatbot's response to the user input
        """
        ########################################################################
        # TODO: Implement the extraction and transformation in this method,    #
        # possibly calling other functions. Although your code is not graded   #
        # directly based on how modular it is, we highly recommended writing   #
        # code in a modular fashion to make it easier to improve and debug.    #
        ########################################################################
        movies = []
        movie_indices = []
        input = Chatbot.preprocess(line)

        # handles user talking about emotions (creative mode feature)

        # building up user matrix of movies
        if (self.movie_count < 5):
            #check if user properly formatted movie in input
            if self.creative == False and not self.check_quotation_marks(line):
                return "You forgot a quotation mark somewhere!!! Try again!"
            
            while("" in input):
                input.remove("")

            movies = Chatbot.extract_titles(self, input)

            if self.creative == True and (len(movies) > 1):
                return "Bert is overwhelmed... Please only tell Bert about one movie at a time!!!"
            elif self.creative == False and (len(movies) > 1):
                return "Please only talk about one movie at a time. Could tell me what you thought about the first movie, please?"

            # if sentiment analysis was wrong the first time, user can correct Bert
            if (self.movie_count != 0) and "no" in input:
                wrong_movie = self.prev_movies[len(self.prev_movies) - 1]
                wrong_movie_index = Chatbot.find_movies_by_title(self, wrong_movie)
                if self.user_ratings[wrong_movie_index] == 1:
                    self.user_ratings[wrong_movie_index] = -1
                else:
                    self.user_ratings[wrong_movie_index] = 1
                if self.creative == True:
                    return "Oh, okay! Bert must've misheard! Bert understands now and has corrected the mistake. (￣ー￣)ゞ Let's talk about a new movie now!"
                elif self.creative == False:
                    return "I'm sorry for the misunderstanding! I've updated your rating of that movie. Feel free to talk about a different movie, now."

            # arbitrary inputs
            if self.creative == True and not movies:
                # check if user expressed their emotions using an expression like "I am (emotion)" or "I feel (emotion)"
                regex = r"(i\s*(am|feel)\s*(\w+))"
                match = re.search(regex, line, re.IGNORECASE)

                if not match:
                    poss_responses = ["Um... are you talking about a movie? Bert's specialty is recommending movies. ╥﹏╥ Remember to put the titles of movies inbetween quotation marks!",
                                  "Bert understands, but Bert wants to talk about movies! (ง •̀_•́)ง‼",
                                  "Oh, Bert sees what you're saying! But Bert really really wants to discuss movies! Tell Bert about a movie! ٩(๑`^´๑)۶",
                                  "Bert only wants to talk about movies though... Tell Bert about a movie! (๑ơ ₃ ơ)♥",
                                  "Bert is getting overwhelmed by all this non-movie talk... let's talk about movies, okay? ٩(๑˃̵ᴗ˂̵๑)۶"]
                    return poss_responses[random.randint(0, len(poss_responses) - 1)]
                elif self.creative == False and not movies:
                    poss_responses = ["Did you forget to include quotation marks around the title? Try again, please.",
                                  "Are we talking about a movie? If so, please remember to put quotation marks around the title.",
                                  "My specialty is talking about movies. If you forgot to put the title in quotation marks, please talk about the movie again and do so.",
                                  "While I do especially enjoy talking to you, I would really love to recommend you a movie. Please talk about a movie, remembering to put the title in quotation marks.",
                                  "Hmm, I can't quite tell if you're talking about a movie. Talk about it again, with the title in quotation marks."]
                    return poss_responses[random.randint(0, len(poss_responses) - 1)]
                else:
                    emotion = match.group(3)
                    return "Oh, Bert thinks you're {}. Do you want to talk to Bert about it? Bert can't make any promises though... ( ˙▿˙ )".format(emotion)
            
            # currently assuming there is only one movie in the list
            for movie in movies:
                if movie not in self.prev_movies:
                    movie = movie.replace('"', "") # removes extra quotation marks
                    movie_indices = Chatbot.find_movies_by_title(self, movie)
                    self.movie_count += 1
                    self.prev_movies.append(movie)
                else:
                    if self.creative == True:
                        return "You already mentioned that movie! Bert's memory isn't that bad! (๑•̀д•́๑) Talk about a different movie!!!"
                    elif self.creative == False:
                        return "I believe you already mentioned that movie. In order to give you the best recommendations possible, please talk about a different movie."
                    
            # if more than one movie found, ask the user to clarify / if no movies are found
            if (len(movie_indices) > 1):
                self.movie_count -= len(movies)
                if self.creative == True:
                    return "Bert found more than one movie with that name! ᕙ(  •̀ ᗜ •́  )ᕗ Bert wants you to put the year in parentheses after the title! Make sure the year is still within the quotation marks, though! (￣ー￣)ゞ"
                else:
                    return "I found more than one movie with that name. Could you clarify by putting the year, or perhaps an alternate title in parantheses afterwards?"
            elif not movie_indices:
                if self.creative == True:
                    return "Oh... Bert doesn't know that movie. (T⌓T) Tell Bert about a different movie, please..."
                else:
                    return "Hmm, I don't seem to know of that movie. I'm sorry, but could you talk about a different movie?"

            sentiment = 0
            
            # currently not configured to handle multiple movies
            for index in movie_indices:
                sentiment = Chatbot.extract_sentiment(self, input)
                if sentiment == 0:
                    self.prev_movies.pop()
                self.user_ratings[index] = sentiment

            # if the user likes the movie
            if sentiment == 1:
                if self.creative == True:
                    possible_positive_responses = ["So you liked {}? That's one of Bert's favorite movies! ᕕ( ᐛ )ᕗ Tell Bert about another movie you've seen.".format(movies[0]),
                                        "Bert sees that you like {}! ∠( ᐛ 」∠)_ Please tell Bert about another movie.".format(movies[0]),
                                        "Bert thinks you like movies like {}. (*･▽･*) Is that right? Tell Bert more!".format(movies[0]),
                                        "{} was an enjoyable movie, wasn't it?!? Tell Bert about a different movie! +･.゜。(´∀｀)。゜.･+".format(movies[0]),
                                        "Bert also thinks {} was a good movie!!! Bert wants to know more about your taste in movies!!! (〜￣▽￣)〜".format(movies[0])]
                    response = possible_positive_responses[random.randint(0, len(possible_positive_responses) - 1)]
                else:
                    possible_positive_responses = ["So you liked {}? I also enjoy that movie! Tell me about another movie you've seen.".format(movies[0]),
                                                   "I see that you like {}. Please tell me about another movie you've seen.".format(movies[0]),
                                                   "I think you like movies like {}, is that right? Tell me about a different movie.".format(movies[0]),
                                                   "{} was an enjoyable movie, wasn't it? Is there another movie you'd like to talk about?".format(movies[0]),
                                                   "I also think {} is a good movie. Can you talk about more movies you like, or perhaps movies you dislike?".format(movies[0])]
                    response = possible_positive_responses[random.randint(0, len(possible_positive_responses) - 1)]
            elif sentiment == -1:
                if self.creative == True:
                    possible_negative_responses = ["Oh no... you didn't like {}? Bert wants to know more about your taste in movies! (๑•﹏•)⋆* ⁑⋆*".format(movies[0]),
                                                "Bert sees that you didn't like {}... ｡ﾟヽ(ﾟ´Д｀)ﾉﾟ｡ Tell Bert about a different movie!".format(movies[0]),
                                                "You don't like movies like {}? Tell Bert about a different movie! Bert needs more information! ヽ(´□｀。)ﾉ".format(movies[0]),
                                                "{} wasn't a good movie, was it? (Bert secretly agrees! ᕕ( ◔3◔)ᕗ) What's your opinion on a different movie?!?".format(movies[0]),
                                                "Bert thinks you don't like {}, is that right?!? ( ✧≖ ͜ʖ≖) Bert is curious about what you think about other movies!!!".format(movies[0])]
                    response = possible_negative_responses[random.randint(0, len(possible_negative_responses) - 1)]
                else:
                    possible_negative_responses = ["Oh, so you didn't like {}? Tell me about more movies, then.".format(movies[0]),
                                                   "I see that you don't like {}. Is there another movie you dislike? Or perhaps one you like?".format(movies[0]),
                                                   "You don't like movies like {}, right? Tell me about a different movie.".format(movies[0]),
                                                   "{} wasn't a good movie, was it? What's your opinion on a different movie?".format(movies[0]),
                                                   "I think you said you don't like {}, is that right? Tell me your opinion on a different movie.".format(movies[0])]
                    response = possible_negative_responses[random.randint(0, len(possible_negative_responses) - 1)]
            elif sentiment == 0:
                self.movie_count -= 1
                if self.creative == True:
                    return "Bert is sorry... Bert doesn't know if you like or dislike that movie... ｡･ﾟﾟ･(>д<)･ﾟﾟ･｡ Tell Bert more about it!!!"
                elif self.creative == False:
                    return "I'm sorry, I can't seem to tell if you like or dislike that movie. Tell me more about it."

        # if user has inputted a sufficient of movies they like or dislike
        if (self.movie_count == 5):
            if self.num_recs_given == 0:
                self.recommendations = Chatbot.recommend(self, self.user_ratings, self.ratings, creative=self.creative)

                for id in self.recommendations:
                    self.recommended_movies.append(self.titles[id][0])

            confirmation_words = ["yes", "yeah", "yea", "yup"]
            # reset global variables
            if self.num_recs_given == (len(self.recommendations) - 1):
                if self.creative == True:
                    response = "This is Bert's last recommendation based on the movies you talked about! Bert really thinks you'll like {}! (･ω<)☆ Type :quit to quit, or talk about more movies for more recommendations!".format(self.recommended_movies.pop(0))
                else:
                    response = "This is my last recommendation based on the movies you talked about. I think you'd enjoy {}. If you'd like more recommendations, feel free to talk about more movies. Otherwise, type ':quit: to quit.".format(self.recommended_movies.pop(0))
                self.user_ratings = np.zeros(self.ratings.shape[0])
                self.prev_movies.clear()
                self.movie_count = 0
                movies.clear()
                movie_indices.clear()
                self.recommendations.clear()
                self.num_recs_given = 0
                self.recommended_movies.clear()
            elif (self.num_recs_given >= 1 and any(word in input for word in confirmation_words)) or self.num_recs_given == 0:
                if self.creative == True:
                    mov = self.recommended_movies.pop(0)
                    res = ["Bert thinks you'll like {}! d(･∀･○) Would you like another recommendation? Otherwise, type :quit to quit!".format(mov),
                           "Try watching {}! Do you want another recommendation~? If not, type ':quit'!".format(mov),
                           "How about watching {}?!? Do you want more recommendations? (๑ơ ₃ ơ)♥ Type :quit if you don't!".format(mov)]
                    response = res[random.randint(0, len(res) - 1)]
                else:
                    mov = self.recommended_movies.pop(0)
                    res = ["I believe you'd enjoy {}. Would you like another recommendation? Otherwise, type ':quit' to quit.".format(mov),
                           "{} seems like a good fit for you. Would you like to hear another recommendation? Type ':quit' to quit".format(mov),
                           "I think you'd like {}. Would you like more recommendations? If you'd like to say goodbye, type ':quit'.".format(mov)]
                    response = res[random.randint(0, len(res) - 1)]
                self.num_recs_given += 1
            else:
                if self.creative == True:
                    response = "Bert hopes you like his movie recommendations!!! You can type ':quit' to quit! (｡･ω･)ﾉﾞ Bye bye~"
                else:
                    response = "I hope you enjoy my recommendations. You can type ':quit' to exit. Goodbye!"
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return response

    @staticmethod
    def preprocess(text):
        """Do any general-purpose pre-processing before extracting information
        from a line of text.

        Given an input line of text, this method should do any general
        pre-processing and return the pre-processed string. The outputs of this
        method will be used as inputs (instead of the original raw text) for the
        extract_titles, extract_sentiment, and extract_sentiment_for_movies
        methods.

        Note that this method is intentially made static, as you shouldn't need
        to use any attributes of Chatbot in this method.

        :param text: a user-supplied line of text
        :returns: the same text, pre-processed
        """
        ########################################################################
        # TODO: Preprocess the text into a desired format.                     #
        # NOTE: This method is completely OPTIONAL. If it is not helpful to    #
        # your implementation to do any generic preprocessing, feel free to    #
        # leave this method unmodified.                                        #
        ########################################################################

        # This splits the text into a list of words, preserving quoted strings
        splitter = shlex.split(text, posix=False)
        
        # This parses through the list of words and stems them, checking first for edge cases
        index = 0
        for word in splitter:
            # Parses strings that do not contain ""
            if "\"" not in word:
                # Removes misc. punctuation and makes all words lowercase
                word = word.translate(str.maketrans('', '', string.punctuation)).lower()
                splitter[index] = word
                if word in Chatbot.edge_cases:
                    #text = text.replace(word, self.edge_cases[word])
                    splitter[index] = Chatbot.edge_cases[word]
            
            # Removed Porter stemmer for now, since it was messing everything up
            # elif "\"" not in word:
            #     #text = text.replace(word, PorterStemmer().stem(word, 0, len(word)-1))
            #     splitter[index] = PorterStemmer().stem(word, 0, len(word)-1)
            index += 1
    
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

        # To return a string instead of a list, uncomment the lines in the loop and change the return statement to text
        return splitter

    def extract_titles(self, preprocessed_input):
        """Extract potential movie titles from a line of pre-processed text.
        
        Given an input text which has been pre-processed with preprocess(),
        this method should return a list of movie titles that are potentially
        in the text.

        - If there are no movie titles in the text, return an empty list.
        - If there is exactly one movie title in the text, return a list
        containing just that one movie title.
        - If there are multiple movie titles in the text, return a list
        of all movie titles you've extracted from the text.

        Example:
          potential_titles = chatbot.extract_titles(chatbot.preprocess(
                                            'I liked "The Notebook" a lot.'))
          print(potential_titles) // prints ["The Notebook"]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: list of movie titles that are potentially in the text
        """
        titles = []
        for word in preprocessed_input:
            if word[0] == "\"":
                titles.append(word.replace('"', "")) # removes extra quotation marks)
        return titles

    def find_movies_by_title(self, title):
        """ Given a movie title, return a list of indices of matching movies.

        - If no movies are found that match the given title, return an empty
        list.
        - If multiple movies are found that match the given title, return a list
        containing all of the indices of these matching movies.
        - If exactly one movie is found that matches the given title, return a
        list
        that contains the index of that matching movie.

        Example:
          ids = chatbot.find_movies_by_title('Titanic')
          print(ids) // prints [1359, 2716]

        :param title: a string containing a movie title
        :returns: a list of indices of matching movies
        """
        ids = []
        articles = ["a", "an", "the", "le", "il", "la", "l'", "i", "le", "les", "un", "une", "des", "du", "el", "los", "las", "una", "unos", "unas"]
        first_word = ""
        first_word_end_index = 0
        end_index = len(title) - 1
        reformatted_title = ""
        regex_year = '([0-9]{4})'

        # Extract first word of title
        for char in title:
            if char == " ":
                break
            else:
                first_word += char
                first_word_end_index += 1
        # Reformat title as "Title Fragment, Article (Year)"
        if first_word.lower() in articles:
            # Article included, year included
            if bool(re.search(regex_year, title)):
                reformatted_title = title[first_word_end_index + 1:end_index - 6] + ", " + first_word + title[end_index - 6:end_index + 1]
            # Article included, year not included
            else:
                reformatted_title = title[first_word_end_index + 1:end_index + 1] + ", " + first_word
        else:
            reformatted_title = title.lower()
        reformatted_title = reformatted_title.lower()
        # Loop through movie data
        for i in range(len(self.titles)):
            official_title = self.titles[i][0].lower()
            # Check if input is exact (since it is unlikely that title is spelled incorrectly, assume this prunes inputs with inexact or nonexistent years)
            if reformatted_title == official_title:
                ids.append(i)
            # Prune for all possible titles containing input substring
            elif reformatted_title in official_title:
                input_start_index = official_title.find(reformatted_title)
                if self.creative:
                    # Alternate title: match "Se7en"
                    if official_title[input_start_index + len(reformatted_title)] == ")":
                        ids.append(i)
                    # Disambiguation 1: prune "Screams" but not "Scream 2"
                    elif not official_title[input_start_index + len(reformatted_title)].isalnum():
                        ids.append(i)
                # Check that wording is (somewhat) exact, i.e. "Scream" is not actually "Scream 2" or "Screams"
                elif official_title[input_start_index + len(reformatted_title) + 1] == "(":
                    ids.append(i)
        return ids

    def extract_sentiment(self, preprocessed_input):
        """Extract a sentiment rating from a line of pre-processed text.

        You should return -1 if the sentiment of the text is negative, 0 if the
        sentiment of the text is neutral (no sentiment detected), or +1 if the
        sentiment of the text is positive.

        As an optional creative extension, return -2 if the sentiment of the
        text is super negative and +2 if the sentiment of the text is super
        positive.

        Example:
          sentiment = chatbot.extract_sentiment(chatbot.preprocess(
                                                    'I liked "The Titanic"'))
          print(sentiment) // prints 1

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a numerical value for the sentiment of the text
        """

        total_pos = 0
        total_neg = 0
        last_word = 0
        negator_present = False

        prev_word = ""
        if self.creative:
            strong_sentiment = False
            emphasis = False            

            for word in preprocessed_input:
                if word in self.negators_conj:
                    negator_present = True
                if word in self.emphasizers:
                    emphasis = True
                if "\"" not in word and word in self.sentiment:
                    if prev_word in self.negators:
                        if word in self.strong_negative:
                            total_pos += 1
                            last_word = 1
                            strong_sentiment = True
                        elif word in self.strong_positive:
                            total_neg += 1
                            last_word = -1
                            strong_sentiment = True
                        elif self.sentiment[word] == 'pos':
                            total_neg += 1
                            last_word = -1
                        else:
                            total_pos += 1
                            last_word = 1
                    else:
                        if word in self.strong_positive:
                            total_pos += 1
                            last_word = 1
                            strong_sentiment = True
                        elif word in self.strong_negative:
                            total_neg += 1
                            last_word = -1
                            strong_sentiment = True
                        elif self.sentiment[word] == 'pos':
                            total_pos += 1
                            last_word = 1
                        else:
                            total_neg += 1
                            last_word = -1               
                if word in self.negators:
                    prev_word = word

            # if there are 1 or more sentiment words per category and a negator is present, do more complex processing
            if negator_present and total_pos >= 1 and total_neg >= 1:
                # if the last word is positive, return 1
                if last_word == 1:
                    return 1 if not (strong_sentiment or emphasis) else 2
                else:
                    return -1 if not (strong_sentiment or emphasis) else -2
            # Catch-all for all other cases: just compare the total number of positive and negative words
            else:
                if total_pos > total_neg:
                    return 1 if not (strong_sentiment or emphasis) else 2
                elif total_pos < total_neg:
                    return -1 if not (strong_sentiment or emphasis) else -2
                else:
                    return 0 
        else:
            for word in preprocessed_input:
                if word in self.negators_conj:
                    negator_present = True
                if "\"" not in word and word in self.sentiment:
                    if prev_word in self.negators:
                        if self.sentiment[word] == 'pos':
                            total_neg += 1
                            last_word = -1
                        else:
                            total_pos += 1
                            last_word = 1
                    else:
                        if self.sentiment[word] == 'pos':
                            total_pos += 1
                            last_word = 1
                        else:
                            total_neg += 1
                            last_word = -1
                if word in self.negators:
                    prev_word = word

            # if there are 1 or more sentiment words per category and a negator is present, do more complex processing
            if negator_present and total_pos >= 1 and total_neg >= 1:
                # if the last word is positive, return 1
                if last_word == 1:
                    return 1
                else:
                    return -1
            # Catch-all for all other cases: just compare the total number of positive and negative words
            else:
                if total_pos > total_neg:
                    return 1
                elif total_pos < total_neg:
                    return -1
                else:
                    return 0                   

    def extract_sentiment_for_movies(self, preprocessed_input):
        """Creative Feature: Extracts the sentiments from a line of
        pre-processed text that may contain multiple movies. Note that the
        sentiments toward the movies may be different.

        You should use the same sentiment values as extract_sentiment, described

        above.
        Hint: feel free to call previously defined functions to implement this.

        Example:
          sentiments = chatbot.extract_sentiment_for_text(
                           chatbot.preprocess(
                           'I liked both "Titanic (1997)" and "Ex Machina".'))
          print(sentiments) // prints [("Titanic (1997)", 1), ("Ex Machina", 1)]

        :param preprocessed_input: a user-supplied line of text that has been
        pre-processed with preprocess()
        :returns: a list of tuples, where the first item in the tuple is a movie
        title, and the second is the sentiment in the text toward that movie
        """
        
        pos_joiner = ['and', 'both', 'either', 'or']
        neg_joiner = ['but', 'neither', 'nor', 'either']
        movies = []
        sublists = []
        final = []

        first_sentiment = 0
        
        # Removes empty strings from the list
        preprocessed_input = [i for i in preprocessed_input if i]

        # Extracts the movies from the preprocessed input and splits the input into sublists
        last_idx = 0
        for word in preprocessed_input:
            if "\"" in word:
                movies.append(word.replace('"', ''))
                sublists.append(preprocessed_input[last_idx:preprocessed_input.index(word) + 1])
                last_idx = preprocessed_input.index(word) + 1
        
        # Extracts the sentiment for the first movie
        first_sentiment = 0
        for word in sublists[0]:
            if word in self.sentiment:
                if prev_word in self.negators:
                    if self.sentiment[word] == 'pos':
                        first_sentiment = -1
                    else:
                        first_sentiment = 1
                else:
                    if self.sentiment[word] == 'pos':
                        first_sentiment = 1
                    else:
                        first_sentiment = -1
            if word not in self.emphasizers:
                prev_word = word            
        final.append((movies[0], first_sentiment))

        # Extracts the sentiment for the rest of the movies, if present
        if len(sublists) > 1:
            for sublist in sublists[1:]:
                if sublist[0] in pos_joiner or "\"" in sublist[0]:
                    final.append((movies[sublists.index(sublist)], first_sentiment))
                elif sublist[0] in neg_joiner:
                    final.append((movies[sublists.index(sublist)], -1 * first_sentiment))
        return final


    def edit_distance(self, str1, str2):
        """Citation: https://leetcode.com/problems/edit-distance/solutions/159295/python-solutions-and-intuition/"""

        m = len(str1)
        n = len(str2)
        table = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(m + 1):
            table[i][0] = i
        for j in range(n + 1):
            table[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if str1[i - 1] == str2[j - 1]:
                    table[i][j] = table[i - 1][j - 1]
                else:
                    table[i][j] = min((1 + table[i - 1][j]), (1 + table[i][j - 1]), (2 + table[i - 1][j - 1]))
        return table[-1][-1]


    def find_movies_closest_to_title(self, title, max_distance=3):
        """Creative Feature: Given a potentially misspelled movie title,
        return a list of the movies in the dataset whose titles have the least
        edit distance from the provided title, and with edit distance at most
        max_distance.

        - If no movies have titles within max_distance of the provided title,
        return an empty list.
        - Otherwise, if there's a movie closer in edit distance to the given
        title than all other movies, return a 1-element list containing its
        index.
        - If there is a tie for closest movie, return a list with the indices
        of all movies tying for minimum edit distance to the given movie.

        Example:
          # should return [1656]
          chatbot.find_movies_closest_to_title("Sleeping Beaty")

        :param title: a potentially misspelled title
        :param max_distance: the maximum edit distance to search for
        :returns: a list of movie indices with titles closest to the given title
        and within edit distance max_distance
        """
        # numpy argmax?
        
        closest = []
        distances = []

        movie = title.lower()
        for t in self.titles:
            index = t[0].find(" (")
            if index != -1:                                           # if there exists a date
                choice = t[0][:index].lower()
            else:
                choice = t[0].lower()
            distances.append(self.edit_distance(movie, choice))
        
        minimum = min(distances)
        for i in range(len(distances)):
            if minimum <= max_distance and distances[i] == minimum:
                closest.append(i)

        return closest


    def disambiguate(self, clarification, candidates):
        """Creative Feature: Given a list of movies that the user could be
        talking about (represented as indices), and a string given by the user
        as clarification (eg. in response to your bot saying "Which movie did
        you mean: Titanic (1953) or Titanic (1997)?"), use the clarification to
        narrow down the list and return a smaller list of candidates (hopefully
        just 1!)

        - If the clarification uniquely identifies one of the movies, this
        should return a 1-element list with the index of that movie.
        - If it's unclear which movie the user means by the clarification, it
        should return a list with the indices it could be referring to (to
        continue the disambiguation dialogue).

        Example:
          chatbot.disambiguate("1997", [1359, 2716]) should return [1359]

        :param clarification: user input intended to disambiguate between the
        given movies
        :param candidates: a list of movie indices
        :returns: a list of indices corresponding to the movies identified by
        the clarification
        """

        clarify = clarification.lower()
        disambiguation = []

        indexes = {1: ["first", "the first one", "number one", "no. 1", "1", "1st"], 2: ["second", "the second one", "number two", "no. 2", "2", "2nd"], 
                   3: ["third", "the third one", "number three", "no. 3", "3", "3rd"], 4: ["fourth", "the fourth one", "number four", "no. 4", "4", "4th"],
                   5: ["five", "the fifth one", "number five", "no. 5", "5", "5th"], 6: ["six", "the sixth one", "number six", "no. 6", "6", "6th"],
                   7: ["seven", "the seventh one", "number seven", "no. 7", "7", "7th"], 8: ["eight", "the eighth one", "number eight", "no. 8", "8", "8th"],
                   9: ["nine", "the ninth one", "number nine", "no. 9", "9", "9th"], 10: ["ten", "the tenth one", "number ten", "no. 10", "10", "10th"]}
        
        recent = ["most recent", "latest", "initial", "prior"]
        old = ["oldest", "least recent", "earliest", "last"]

        year = "[0-9]{4}"
        id = "the ([A-Za-z ]+) one"
        the_one = "asdflker;jew;qr"
        
        # checking if clarify is talking about first, second, third, etc.
        for index in indexes:
            if clarify in indexes[index]:
                disambiguation.append(candidates[index - 1])

        # checks if clarify talks about first or last movie made
        if clarify in recent:
            disambiguation.append(candidates[0])
        elif clarify in old:
            disambiguation.append(candidates[len(candidates)])
        
        # if clarify has the...one, check if it talks about first/last movie
        # or if capture group exists inside the title
        elif bool(re.search(id, clarify)) == True:
            the_one = re.search(id, clarify).group(1).lower()
            if the_one in recent:
                disambiguation.append(candidates[0])
            elif the_one in old:
                disambiguation.append(candidates[len(candidates)])
            else:
                for c in candidates:
                    i = self.titles[c][0].find(" (")
                    choice = self.titles[c][0][:i].lower()
                    if the_one in choice:
                        disambiguation.append(c)
        
        # if user specifies year
        elif bool(re.search(year, clarification)) == True:
            for c in candidates:
                title = self.titles[c][0]
                if clarification in title:
                    disambiguation.append(c)

        # if clarification exists inside title
        else:
            for c in candidates:
                i = self.titles[c][0].find(" (")
                choice = self.titles[c][0][:i].lower()
                if clarify in choice:           
                    disambiguation.append(c)

        return disambiguation


    ############################################################################
    # 3. Movie Recommendation helper functions                                 #
    ############################################################################

    @staticmethod
    def binarize(ratings, threshold=2.5):
        """Return a binarized version of the given matrix.

        To binarize a matrix, replace all entries above the threshold with 1.
        and replace all entries at or below the threshold with a -1.

        Entries whose values are 0 represent null values and should remain at 0.

        Note that this method is intentionally made static, as you shouldn't use
        any attributes of Chatbot like self.ratings in this method.

        :param ratings: a (num_movies x num_users) matrix of user ratings, from
         0.5 to 5.0
        :param threshold: Numerical rating above which ratings are considered
        positive

        :returns: a binarized version of the movie-rating matrix
        """
        # do not use self.ratings directly in this function.
        new_ratings = ratings

        new_ratings[(ratings <= threshold) & (ratings != 0)] = -1.0
        new_ratings[(ratings > threshold) & (ratings != 0)] = 1.0

        binarized_ratings = new_ratings

        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """

        numerator = np.dot(u, v)
        u_norm = 0 + 1e-30
        v_norm = 0 + 1e-30
        
        for w in u:
            u_norm += w ** 2
        u_norm = math.sqrt(u_norm)
        
        for w in v:
            v_norm += w ** 2
        v_norm = math.sqrt(v_norm)
        
        cosine_sim = numerator / (u_norm * v_norm)
    
        return cosine_sim

    def recommend(self, user_ratings, ratings_matrix, k=10, creative=False):
        """Generate a list of indices of movies to recommend using collaborative
         filtering.

        You should return a collection of `k` indices of movies recommendations.

        As a precondition, user_ratings and ratings_matrix are both binarized.

        Remember to exclude movies the user has already rated!

        Please do not use self.ratings directly in this method.

        :param user_ratings: a binarized 1D numpy array of the user's movie
            ratings
        :param ratings_matrix: a binarized 2D numpy matrix of all ratings, where
          `ratings_matrix[i, j]` is the rating for movie i by user j
        :param k: the number of recommendations to generate
        :param creative: whether the chatbot is in creative mode

        :returns: a list of k movie indices corresponding to movies in
        ratings_matrix, in descending order of recommendation.
        """

        # item-item collaborative filtering with cosine similarity, no mean-centering, and no normalization of scores.                                                              #
        # exclude movies the user has already rated
        # can assume ratings_matrix does not contain the current user's ratings.

        scores = []
        recommendations = []

        for i in range(len(user_ratings)):
            if user_ratings[i] == 0:                    # if there is no rating yet, grab vector for it
                unwatched_vector = ratings_matrix[i]

                sum = 0
                for j in range(len(ratings_matrix)):    # go through all movies
                    if user_ratings[j] != 0:            # if a particular movie has been watched, compute cosine sim and add it * weight
                        sum += user_ratings[j] * self.similarity(unwatched_vector, ratings_matrix[j])
                scores.append(sum)

            else:
                scores.append(0)                        # now, we've built a list of size user_ratings w ratings for each movie

        scores = np.argsort(scores)

        indexes = scores[-k:]
        for index in indexes:
            recommendations.append(index)
        
        recommendations.reverse()

        return recommendations

    ############################################################################
    # 4. Debug info                                                            #
    ############################################################################

    def debug(self, line):
        """
        Return debug information as a string for the line string from the REPL

        NOTE: Pass the debug information that you may think is important for
        your evaluators.
        """
        debug_info = 'debug info'
        return debug_info

    ############################################################################
    # 5. Write a description for your chatbot here!                            #
    ############################################################################
    def intro(self):
        """Return a string to use as your chatbot's description for the user.

        Consider adding to this description any information about what your
        chatbot can do and how the user can interact with it.
        """
        return """
        Bert is a movie recommender bot who can give you movie recommendations based on your personal taste. After you 
        talk about 5 movies (at minimum), Bert will start giving you movie recommendations. 
        Make sure to include the title of the movie in quotes. If you need to correct Bert's evaluation of your opinion of
        a movie, you can do so. 
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
