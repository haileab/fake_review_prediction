#!/usr/bin/env python

import math

from utils import get_char_count
from utils import get_words
from utils import get_sentences
from utils import count_syllables
from utils import count_complex_words


class Readability:
    analyzedVars = {}

    def __init__(self, text):
        self.analyze_text(text)

    def analyze_text(self, text):
        words = get_words(text)
        char_count = get_char_count(words)
        word_count = len(words)
        sentence_count = len(get_sentences(text))
        syllable_count = count_syllables(words)
        complexwords_count = count_complex_words(text)
        avg_words_p_sentence = word_count/sentence_count

        self.analyzedVars = {
            'words': words,
            'char_cnt': float(char_count),
            'word_cnt': float(word_count),
            'sentence_cnt': float(sentence_count),
            'syllable_cnt': float(syllable_count),
            'complex_word_cnt': float(complexwords_count),
            'avg_words_p_sentence': float(avg_words_p_sentence)
        }

    def ARI(self):
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            score = 4.71 * (self.analyzedVars['char_cnt'] / self.analyzedVars['word_cnt']) + 0.5 * (self.analyzedVars['word_cnt'] / self.analyzedVars['sentence_cnt']) - 21.43
        return score

    def FleschReadingEase(self):
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            score = 206.835 - (1.015 * (self.analyzedVars['avg_words_p_sentence'])) - (84.6 * (self.analyzedVars['syllable_cnt']/ self.analyzedVars['word_cnt']))
        return round(score, 4)

    def FleschKincaidGradeLevel(self):
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            score = 0.39 * (self.analyzedVars['avg_words_p_sentence']) + 11.8 * (self.analyzedVars['syllable_cnt']/ self.analyzedVars['word_cnt']) - 15.59
        return round(score, 4)

    def GunningFogIndex(self):
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            score = 0.4 * ((self.analyzedVars['avg_words_p_sentence']) + (100 * (self.analyzedVars['complex_word_cnt']/self.analyzedVars['word_cnt'])))
        return round(score, 4)

    def SMOGIndex(self):
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            score = (math.sqrt(self.analyzedVars['complex_word_cnt']*(30/self.analyzedVars['sentence_cnt'])) + 3)
        return score

    def ColemanLiauIndex(self):
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            score = (5.89*(self.analyzedVars['char_cnt']/self.analyzedVars['word_cnt']))-(30*(self.analyzedVars['sentence_cnt']/self.analyzedVars['word_cnt']))-15.8
        return round(score, 4)

    def LIX(self):
        longwords = 0.0
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            for word in self.analyzedVars['words']:
                if len(word) >= 7:
                    longwords += 1.0
            score = self.analyzedVars['word_cnt'] / self.analyzedVars['sentence_cnt'] + float(100 * longwords) / self.analyzedVars['word_cnt']
        return score

    def RIX(self):
        longwords = 0.0
        score = 0.0
        if self.analyzedVars['word_cnt'] > 0.0:
            for word in self.analyzedVars['words']:
                if len(word) >= 7:
                    longwords += 1.0
            score = longwords / self.analyzedVars['sentence_cnt']
        return score


# if __name__ == "__main__":
#     text = """Take a bottle of wine, order the mussels, soak the french fries and fresh crusty bread in the garlicky wine broth, and save room for the eponymous dessert.  This adorable little closet of a byo French restaurant with its inexplicable lighthouse-themed decor is well worth the walk through the cutest part of the West Village (closest subway stations are 14th St. on the 8 AV lines, 8 AV on the L, and Chistopher St. on the 1/9) and it's even worth the inevitable wait between 7pm and 9pm (add an hour in each direction on the weekends, and I'd say forget entirely about getting a table on gorgeous summer nights except that it's hard to imagine better patio dining).  The brisk waitstaff will typically offer to uncork your bottle for you to enjoy while you wait for a table.  If you're not into mussels, the menu offers ligher fare like croque monsieur, custom omlettes and tarte du jour as well as classic French appetizers and entrees like escargots, a roasted pear salad, chicken baked in a puff pastry, salmon with julienned vegetables in a beurre blanc, or beef mignonette with french fries.  It probably won't be the very best food you've ever had, but it's equally unlikely to disappoint, and with entrees priced in the $12-$17 range, it could be one of the best deals you'll find in Manhattan and will certainly be one of the most charming. """
#
#     rd = Readability(text)
#     print( 'Test text:')
#     print( '"%s"\n' % text)
#     print( 'ARI: ', rd.ARI())
#     print( 'FleschReadingEase: ', rd.FleschReadingEase())
#     print( 'FleschKincaidGradeLevel: ', rd.FleschKincaidGradeLevel())
#     print( 'GunningFogIndex: ', rd.GunningFogIndex())
#     print( 'SMOGIndex: ', rd.SMOGIndex())
#     print( 'ColemanLiauIndex: ', rd.ColemanLiauIndex())
#     print( 'LIX: ', rd.LIX())
#     print( 'RIX: ', rd.RIX())
