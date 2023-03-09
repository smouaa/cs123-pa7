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
from deps import nltk
from porter_stemmer import PorterStemmer
import random


# noinspection PyMethodMayBeStatic
class Chatbot:
    """Simple class to implement the chatbot for PA 6."""

    edge_cases = {'enjoyed': 'enjoy', 'enjoying': 'enjoy', 'enjoys': 'enjoy', 'loved': 'love', 'loving': 'love', 'loves': 'love', 'hated': 'hate', 'hates': 'hate', 'hating': 'hate', 'disliked': 'dislike', 'dislikes': 'dislike', 'disliking': 'dislike', 'liked': 'like', 'likes': 'like', 'liking': 'like'}

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
        

        # print('I loved "10 things I hate about you": ', self.extract_sentiment(self.preprocess('I loved "10 things I hate about you"'))) 
        # print('I hated "10 things I hate about you", but I loved it: ', self.extract_sentiment(self.preprocess('I hated "10 things I hate about you", but I loved it'))) 
        # print('"Titanic (1997)" started out terrible, but the ending was totally great and I loved it!: ', self.extract_sentiment(self.preprocess('"Titanic (1997)" started out terrible, but the ending was totally great and I loved it!'))) 
        # print('I thought "Titanic (1997)" was great at first, but I then hated it: ', self.extract_sentiment(self.preprocess('I thought "Titanic (1997)" was great at first, but I then hated it')))
        # print('I watched "Titanic (1997)" but thought nothing of it: ', self.extract_sentiment(self.preprocess('I watched "Titanic (1997)" but thought nothing of it')))
        # print('I watched "Titanic (1997)". Hate love hated loved: ', self.extract_sentiment(self.preprocess('I watched "Titanic (1997)". Hate love hated loved')))
        # print('I did not like "Titanic (1997)": ', self.extract_sentiment(self.preprocess('I did not like "Titanic (1997)"')))
        # print("I didn't like \"Titanic (1997)\" at all: ", self.extract_sentiment(self.preprocess("I didn't like \"Titanic (1997)\" at all")))

        # print(self.extract_titles(self.preprocess('I liked "The Notebook" a lot.')))
        # print('Titanic: ', self.find_movies_by_title('Titanic'))
        
        # count number of movies that the user has rated
        self.movie_count = 0

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

        greeting_message = "Hi! I'm Bert. Tell me about a movie you like or dislike. If possible, put the title of the movie in quotation marks."

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

        goodbye_message = "Goodbye!"

        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
        return goodbye_message

    ############################################################################
    # 2. Modules 2 and 3: extraction and transformation                        #
    ############################################################################

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
        #if self.creative:
        #    response = "I processed {} in creative mode!!".format(line)
        #else:
        #    response = "I processed {} in starter mode!!".format(line)

        ######################### PLACEHOLDER (if user corrects sentiment)
        reponse = ""
        movies = []
        movies_indices = []
        if (self.movie_count < 5):
            input = Chatbot.preprocess(line)
            movies = Chatbot.extract_titles(self, input)
            # if user doesn't talk about movies or if no movie titles are found
            if not movies:
                return "Um... are you talking about a movie? I want to talk about movies only."
            # currently assuming there is only one movie in the list
            for movie in movies:
                movie = movie.replace('"', "") # removes extra quotation marks
                movie_indices = Chatbot.find_movies_by_title(self, movie)
                self.movie_count += 1

            # if more than one movie found, ask the user to clarify
            if (len(movie_indices) > 1):
                self.movie_count -= len(movies)
                return "I found more than one movie with that name. Can you try specifying the year in parentheses or checking to see if it's spelled correctly?"
            elif not movie_indices:
                return "I'm sorry, I don't know that movie. I'm kinda old."
            sentiment = 0
            
            # currently not configured to handle multiple movies
            for index in movie_indices:
                sentiment = Chatbot.extract_sentiment(self, input)
                self.user_ratings[index] = sentiment

            # if the user likes the movie
            if sentiment == 1:
                possible_positive_responses = ["So you liked {}? Tell me about another movie you've seen.".format(movies[0]),
                                      "I see that you enjoyed watching {}. Please tell me about another movie.".format(movies[0]),
                                      "You like movies like {}, correct? Tell me about another movie, please.".format(movies[0]),
                                      "{} was an enjoyable movie, wasn't it? Tell me about a different movie.".format(movies[0])]
                response = possible_positive_responses[random.randint(0, len(possible_positive_responses) - 1)]
            elif sentiment == -1:
                possible_negative_responses = ["So you didn't like {}? Tell me your opinion on another movie, please.".format(movies[0]),
                                               "I see that you didn't like {}? Tell me about a different movie.".format(movies[0]),
                                               "You don't like movies like {}, right? Tell me about a different movie, perhaps one that you like.".format(movies[0]),
                                               "{} wasn't a good movie, was it? What's your opinion on a different movie?".format(movies[0])]
                response = possible_negative_responses[random.randint(0, len(possible_negative_responses) - 1)]
            elif sentiment == 0:
                self.movie_count -= 1
                return "I'm sorry, but I'm not sure if you like or dislike that movie. Tell me more about it."

        if (self.movie_count == 5):
            recommendations = Chatbot.recommend(self, self.user_ratings, self.ratings, creative=self.creative)
            recommended_movies = []

            for id in recommendations:
                recommended_movies.append(self.titles[id][0])
            response = f"Here is a list of movies I think you'll like: {', '.join([item for item in recommended_movies[:-1]])} and {recommended_movies[-1]}. If you'd like more recommendations, tell me about more movies."
            self.user_ratings = np.zeros(self.ratings.shape[0])
            self.movie_count = 0

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
        articles = ["a", "an", "the"]
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
            # Check if input is exact
            if reformatted_title == official_title:
                ids.append(i)
            # Prune for all possible titles containing input substring
            elif reformatted_title in official_title:
                input_start_index = official_title.find(reformatted_title)
                # Check that wording is (somewhat) exact, i.e. "Scream" is not actually "Scream 2" or "Screaming"
                if official_title[input_start_index + len(reformatted_title) + 1] == "(":
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

        # List of words that negate the sentiment of the phrase
        # e.g. "Titanic (1997)" started out terrible, but the ending was totally great and I loved it!" -> 1
        #print(preprocessed_input)
        negators = ['not', 'didnt', 'never', 'cant']
        negators_conj = ['but', 'yet', 'nonetheless', 'although', 'despite', 'however', 'nevertheless', 'still', 'though', 'unless', 'unlike', 'until', 'whereas']
        emphasizers = ['really', 'very', 'extremely', 'totally', 'completely', 'absolutely', 'utterly', 'perfectly', 'entirely', 'thoroughly', 'completely', 'utterly', 'fully', 'wholly', 'altogether', 'entirely', 'fully', 'perfectly', 'quite', 'rather', 'somewhat', 'too', 'utterly', 'very', 'awfully', 'badly', 'completely', 'considerably', 'decidedly', 'deeply', 'enormously', 'entirely', 'especially', 'exceptionally', 'extremely', 'fiercely', 'flipping']

        total_pos = 0
        total_neg = 0
        last_word = 0
        negator_present = False

        index = 0
        prev_word = ""
        for word in preprocessed_input: # <---- this is currently iterating through each char, not each word
            if word in negators_conj:
                negator_present = True
            if "\"" not in word and word in self.sentiment:
                if prev_word in negators:
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

            index += 1
            if word not in emphasizers:
                prev_word = word

        #print("Total Positive: ", total_pos)
        #print("Total Negative: ", total_neg)
        #print("Negator Present: ", negator_present)

        # if the total number of sentiment words is 1 or less, do basic processing
        if total_pos + total_neg < 2:
            if total_pos > total_neg:
                return 1
            elif total_pos < total_neg:
                return -1
            else:
                return 0
        # if there are 1 or more sentiment words per category and a negator is present, do more complex processing
        elif negator_present and total_pos >= 1 and total_neg >= 1:
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
        pass

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

        pass

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
        pass

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
        ########################################################################
        # TODO: Binarize the supplied ratings matrix.                          #
        #                                                                      #
        # WARNING: Do not use self.ratings directly in this function.          #
        ########################################################################

        # The starter code returns a new matrix shaped like ratings but full of
        # zeros.
        
        # ratings = num_movies x num_users matrix of user ratings, from 0.5 to 5.0
        # threshold = numerical rating above which ratings are considered positive
        # returns = binarized version of the matrix

        new_ratings = ratings

        new_ratings[(ratings <= threshold) & (ratings != 0)] = -1.0
        new_ratings[(ratings > threshold) & (ratings != 0)] = 1.0


        binarized_ratings = new_ratings

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
        return binarized_ratings

    def similarity(self, u, v):
        """Calculate the cosine similarity between two vectors.

        You may assume that the two arguments have the same shape.

        :param u: one vector, as a 1D numpy array
        :param v: another vector, as a 1D numpy array

        :returns: the cosine similarity between the two vectors
        """
        ########################################################################
        # TODO: Compute cosine similarity between the two vectors.             #
        ########################################################################

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
    
        ########################################################################
        #                          END OF YOUR CODE                            #
        ########################################################################
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

        ########################################################################
        # TODO: Implement a recommendation function that takes a vector        #
        # user_ratings and matrix ratings_matrix and outputs a list of movies  #
        # recommended by the chatbot.                                          #
        #                                                                      #
        # WARNING: Do not use the self.ratings matrix directly in this         #
        # function.                                                            #
        #                                                                      #
        # For starter mode, you should use item-item collaborative filtering   #
        # with cosine similarity, no mean-centering, and no normalization of   #
        # scores.                                                              #
        ########################################################################

        # exclude movies the user has already rated

        # ratings_matrix - binarized 2D numpy matrix of all ratings, where ratings_matrix[i, j] is the rating for movie i by user j
        # input the provided vector of the user's preferences and a pre-processed matrix of ratings by other users 
        # can assume ratings_matrix does not contain the current user's ratings.


        # for each movie i in the dataset
        # calculate the rating of user's rating of the movie i and the cosine between vectors for movies i and j

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

        ########################################################################
        #                        END OF YOUR CODE                              #
        ########################################################################
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
        Your task is to implement the chatbot as detailed in the PA7
        instructions.
        Remember: in the starter mode, movie names will come in quotation marks
        and expressions of sentiment will be simple!
        TODO: Write here the description for your own chatbot!
        """


if __name__ == '__main__':
    print('To run your chatbot in an interactive loop from the command line, '
          'run:')
    print('    python3 repl.py')
