from __future__ import division
import numpy as np
import random
import csv
import re
import string
import unicodedata
import operator
import pickle
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from collections import Counter
import sets
import os
import errno

from gensim import corpora, models, similarities
import gensim
import logging

#-------------------------
# DATA CLEAN UP
#-------------------------

def clean_up_data( matrix ):
    # clean up the matrix
    clean = [ clean_up( row[:] ) for row in matrix ]
    return clean

def clean_up_data_no_prune( matrix ):
    matrix = matrix[1:]
    # clean up the matrix
    clean = [ clean_up_no_prune( row[:] ) for row in matrix ]
    return clean

# helper
# wrapper for all the preprocessing functions
def clean_up(matrix_row):
    fresh = remove_words(     # Remove words like an, is, the, etc.
            plurals(          # Convert plural words into singular
            remove_dot(       # Remove periods at the end of words
            str.split(        # Split the string into words, space as a separator
            conv_uni(         # Convert from unicode to string
            remove_punc(      # Remove punctuations like ",!?.  
            strip_html(       # Convert html special characters into equivalent form, instead of acii (might be redundant with remove_func on top)
            merge_rows(       # Merge rows just in case they're separated by the csv reader( for now, one row is one feedback of one student)  
              matrix_row )))) # The actual row
            .lower()))))       # Convert everything into lower case

    #print fresh
    return fresh

# helper
# wrapper for all the preprocessing functions
def clean_up_no_prune(matrix_row):
    fresh = remove_dot(
            plurals(          # Convert plural words into singular
            str.split(        # Split the string into words, space as a separator
            conv_uni(         # Convert from unicode to string
            remove_punc(      # Remove punctuations like ",!?.  
            strip_html(       # Convert html special characters into equivalent form, instead of acii (might be redundant with remove_func on top)
            merge_rows(       # Merge rows just in case they're separated by the csv reader( for now, one row is one feedback of one student)  
              matrix_row )))) # The actual row
            .lower())))      # Convert everything into lower case  
    return fresh

# helper
# convert unicode to string
# if string already, return that
def conv_uni(unicode_row):
    if isinstance(unicode_row, str):
        return unicode_row
    clean = unicodedata.normalize('NFKD', unicode_row).encode('ascii','ignore')
    return clean

# helper
# merge rows in the matrix
# e.g. [ 'hello', 'hi' ] will be [ 'hello hi' ]
def merge_rows(matrix_row):
    t = ''
    for i in matrix_row:
        t += i
    return t

def plurals( words ):
    lmtzr = WordNetLemmatizer()
    new_words = [ lmtzr.lemmatize(elem) for elem in words ]
    return new_words

def remove_words( words ):
    extra = ['explain', 'equation', 'screencast', 'difficult', 'video', 'understand',
             'understanding', 'doe', 'bit']
    dolch220 = ['a', 'all', 'after', 'always', 'about', 'and', 'am', 'again',
                'around', 'better', 'away', 'are', 'an', 'because', 'bring',
                'bat', 'any', 'been', 'carry', 'blue', 'ate', 'as', 'before',
                'can', 'be', 'ask', 'best', 'cut', 'come', 'by', 'both', 'done',
                'down', 'could', 'buy', 'draw', 'find', 'but', 'every', 'call',
                'drink', 'for', 'came', 'funny', 'did', 'from', 'does', 'fall',
                'go', 'do', 'give', "don't", 'far', 'dont', 'help', 'eat', 'going',
                'full', 'here', 'had', 'got', 'I', 'get', 'has', 'grow', 'in',
                'good', 'her', 'found', 'hold', 'is', 'have', 'him', 'gave',
                'it', 'he', 'his', 'goes', 'hurt', 'jump', 'into', 'how', 'if',
                'like', 'just', 'its', 'keep', 'look', 'must', 'know', 'made',
                'kind', 'make', 'new', 'let', 'many', 'laugh', 'me', 'no',
                'live', 'off', 'my', 'now', 'may', 'or', 'long', 'not', 'on',
                'of', 'much', 'our', 'old', 'read', 'myself', 'play', 'out',
                'once', 'right', 'never', 'please', 'open', 'sing', 'only',
                'run', 'pretty', 'over', 'sit', 'own', 'said', 'ran', 'put',
                'sleep', 'pick', 'see', 'ride', 'round', 'tell', 'the', 'saw',
                'some', 'their', 'shall', 'say', 'stop', 'these', 'show',
                'to', 'she', 'take', 'those', 'so', 'thank', 'upon', 'up',
                'soon', 'them', 'us', 'start', 'we', 'that', 'then', 'use',
                'where', 'there', 'think', 'very', 'today', 'they', 'walk',
                'wash', 'together', 'you', 'this', 'were', 'which', 'try',
                'too', 'when', 'why', 'under', 'wa', 'wish', 'want', 'work', 'was',
                'would', 'well', 'write', 'went', 'your', 'what', 'white',
                'hill', 'who', 'will', 'with', 'yes']    
    words_to_remove = stopwords.words('english') + dolch220 + extra
    return filter(lambda x: x not in words_to_remove, words)

def remove_dot( words ):
    new_words=[]
    for word in words:
        if word.endswith("."):
            new_words += [word[:-1]]
        else:
            new_words += [word]
    return new_words

# helper
# remove punctuations for easy analysis
def remove_punc(sentence):
    # pad punctuations with space
    # just in case so 2 words wont stick together
    # e.g. avoid this: "by this!Also" --> "by thisAlso"
    
    sentence = re.sub('([,!?()])', r' \1 ', sentence)
    # get rid of punctuations
    exclude = set(string.punctuation)

    # don't remove these two symbols
    exclude.remove('-')
    exclude.remove('_')
    exclude.remove('.')

    # remove the punctuations
    sentence = ''.join(ch for ch in sentence if ch not in exclude)
    return sentence

# helper
# This function is by Fredrik Lundh
# convert html special characters
def strip_html(text):
    def fixup(m):
        text = m.group(0)
        if text[:1] == "<":
            return "" # ignore tags
        if text[:2] == "&#":
            try:
                if text[:3] == "&#x":
                    return unichr(int(text[3:-1], 16))
                else:
                    return unichr(int(text[2:-1]))
            except ValueError:
                pass
        elif text[:1] == "&":
            import htmlentitydefs
            entity = htmlentitydefs.entitydefs.get(text[1:-1])
            if entity:
                if entity[:2] == "&#":
                    try:
                        return unichr(int(entity[2:-1]))
                    except ValueError:
                        pass
                else:
                    return unicode(entity, "iso-8859-1")
        return text # leave as is
    return re.sub("(?s)<[^>]*>|&#?\w+;", fixup, text)



#-------------------------
# COUNTING
#-------------------------#

def count_all_words(matrix):
    c1 = Counter()
    for row in matrix:
        c1 += Counter(row)
    return c1

def count_student_words( words, blankdictionary ):
    x = dict(blankdictionary)
    for word in words:
        if word in x:
            x[word] = x[word] + 1
    return x

def dict_to_array( dictionary ):
    x = sort_by_key( dictionary )
    x = np.array( x )#, dtype = [('y', '|S11'), ('Value', float)] )
    return x
    
def matrix_of_Counts(matrix, blank):
    c1 = Counter()
    for row in matrix:  
        c1 += Counter(row)
    return c1

def recreate_wordle_matrix( dictionary ):
    new_list = []
    for word in dictionary.keys():
        for i in range(int(round(dictionary.get(word)))):
            new_list += [word]
    return new_list

def recreate_wordle_matrix_from_array( array ):
    new_list = []
    for word in c3.keys():
        for i in range(c3.get(word)):
            new_list += [word]
    return new_list

def set_dict_values_to_zero( dictionary ):
    x = dict(dictionary)
    for key in x.keys():
        x[key] = 0
    return x

def set_minimum(dictionary, minimum):
    x = dict(dictionary)
    for key in x.keys():
        if x[key] <= minimum:
            del x[key]
    return x

def set_threshold( matrix, dictionary, minimum ):
    return [ set_minimum(row, dictionary, minimum) for row in matrix ]

def sort_by_key(unsorted):
    sorted_tuple = sorted(unsorted.items(), key=operator.itemgetter(0), reverse=False)
    return sorted_tuple

def sort_by_value(unsorted):
    sorted_tuple = sorted(unsorted.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_tuple

# constuct the word matrix
# only accepts a 2D student comment matrix
def word_matrix( student_comment_matrix, dictionary ):
    z = []
    for student_comment in student_comment_matrix:
        a = count_student_words( student_comment, dictionary )
        # converts the dictionary count into an array, 
        # sorts the words as well
        a = dict_to_array(a)
        a = map(int, a[:,1])
        z = z+[a]
    x = np.array(z)
    return x


#-------------------------
# FILE IO
#-------------------------

def print_to_file( name, matrix ):
    directory = 'Results'
    make_sure_path_exists(directory)
    new_path = directory + '/' + name
    f1=open(new_path, 'w+')
    for row in matrix:
        print >>f1, row
    f1.close()

def make_sure_path_exists(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
    
def save_matrix_to_file( name, matrix ):
    save_file( name, matrix )

def load_matrix_from_file( name ):
    return load_file( name )

def save_dict_file( filename, dictionary ):
    save_file( filename, dictionary )

def open_dict_file( filename ):
    return load_file( filename )

def save_words_file( filename, words ):
    save_file( filename, words )

def open_words_file( filename ):
    return load_file( filename )

def save_file( filename, data ):
    f = open( filename, 'wb' ) #b means binary
    pickle.dump( data, f)
    f.close()

def load_file( filename ):
    f = open( filename, 'rb' )
    data = pickle.load( f )
    f.close()
    return data

def read_answer_log( filename, quizname, numbers ):

    answer_log = csv.reader(open(filename, 'rb'), delimiter='\t', quoting=csv.QUOTE_NONE)
    # convert to a list
    y = list(answer_log)
    # then convert to a numpy matrix
    matrix2 = np.array(y)

    matrix_joined = []
    x = 0
    last, = matrix2.shape
    last = last - 1
    
    while x < last :
        f = matrix2[x]
        matrix_joined = matrix_joined + [f]

        y = x+1
        if y > last:
            break

        while not has_time_stamp( matrix2[y] ):
            tol = matrix_joined[-1]
            lol = matrix2[y]
            matrix_joined[-1] = tol + lol
            y = y + 1
            if y > last:
                break
        # end while loop

        x = y
        if y > last:
            break
    # end while loop

    # only retrieve the test we want
    harvest = [ row for row in matrix_joined if get_test_name_from_line( row ) == quizname]
    # only get the numbers we specified 
    harvest = [ row for row in harvest if get_number_from_line( row ) in [1] ]
    harvest = np.array( harvest)
    harvest = remove_duplicate_submits( harvest, quizname, numbers)

    name_list  = np.array([ get_name_from_line(row) for row in harvest ])
    essay_list = np.array([ get_essay_from_line(row) for row in harvest ])

    dual_list = ( zip( name_list, essay_list ) )

    print_to_file( 'essay_with_names.csv', dual_list)
    print_to_file( 'essay.csv', essay_list)

    harvest = np.array( [ row for row in essay_list if row != '' ] )
    print_to_file( 'noblanks.csv', harvest)

    return harvest

# helper function
# this ensures that the last submitted answer
# from a student is the one we extract
def remove_duplicate_submits( matrix, quizname, numbers ):
    names = set([])

    # add names to the set
    # the set doesn't add if the name
    # is in it already
    for row in matrix:
        name = get_name_from_line(row)
        names.add(name)
    latest_stamp = 0
    
    # for each name in the set
    # get the latest time stamp
    # then filter out the earlier ones
    while True and len(names) > 0:
        curname = names.pop()
        submatrix = np.array([ row for row in matrix if get_name_from_line(row) == curname])
        # get latest time stamp
        for row in submatrix:
            stamp = get_time_stamp_from_line(row)
            if latest_stamp < stamp:
                latest_stamp = stamp

        # remove duplicates
        matrix = [ row for row in matrix if (curname != get_name_from_line(row)) or (get_time_stamp_from_line(row) == latest_stamp) ]       
        latest_stamp = 0
        if len(names) == 0:
            break
    # abi, TODO
    # simple fix for now
    # remove the guy with this ID
    matrix =  [ row for row in matrix if get_name_from_line(row) != 'ITP1MGS34Q06' ]
    matrix = np.array(matrix)
    return matrix

def get_essay_from_line( row ):
    # first index of
    # essay answer is 2
    index = 2
    last, = np.array(row).shape
    last  = last - 1 

    answer = str.split(row[index], '[')
    
    index  = index + 1
    while index <= last:
        if len(str.split(row[index], ' ')) < 2:
            break
        answer = answer + [row[index]]
        index  = index + 1     
    answer = ' '.join(answer)
    return answer

def get_name_from_line( row ):
    return str.split(str.split(row[0], ']')[1], '|' )[1]

def get_test_name_from_line( row ):
    f = np.array(row)
    return str.split(str.split(row[0], ']')[1], '|' )[2]

def get_time_stamp_from_line( row ):
    return str.split(row[1], ']')

def get_number_from_line( row ):
    return int(str.split(str.split(row[0], ']')[1], '|' )[3])

# Check if this string is a time stamp
# Stamps look like this :
# [Tue Dec 18 15:03:25 2012] |bb_demo_17032|Lecture_1|2|010101
# This just checks if the string has the correct number of elements
def has_time_stamp( string ):
    # check if empty first
    if string == '':
        return False
    if string == None:
        return False
    if string == []:
        return False
    
    # only consider the first element
    # since that is where the time stamp is
    local = string[0]
    local = np.array(local.split())
    
    outer, = local.shape
    if outer != 6: 
        return False
    
    inner, = np.array(local[5].replace('|', ' ').split()).shape
    if inner != 4:
        return False

    # outer has 6 elements
    # inner has 4 elements
    return True
    
#-------------------------
# WEIGHTS
#-------------------------

# matrix must be a numpy 2D array
def local_log_weighting( matrix ):
    A = matrix
    (rows,_) = A.shape
    for i in range(rows):
        A[i] = np.log2(A[i]+1)
    return A

# matrix must be a numpy 2D array
def local_aug_weighting( matrix ):
    A = matrix
    (rows,_) = A.shape
    y = 0
    # this is array wise math
    for i in range(rows):
        m = A[i].max(axis=0)
        A[i] = ((A[i]/m)+1) / 2
    return A

# matrix must be a numpy 2D array
def local_binary_weighting( matrix ):
    A = matrix
    (rows,columns) = A.shape
    y = 0
    for i in range(rows):
        for j in range(columns):
            if A[i][j] != 0:
                A[i][j] = 1                
    return A

def global_normal_weighting( matrix ):
    # for each word, square the values then sum them
    # return the inverse of that 
    (x,_) = matrix.shape
    #print matrix.shape
    g = np.zeros(x)

    for i in range(x):
        g[i] = 1/np.sqrt(np.sum((matrix[i])**2))

    return g

def global_gfldf_weighting( matrix ):
    # for each word, square the values then sum them
    # return the inverse of that 
    (x,_) = matrix.shape
    g = np.zeros(x)

    for i in range(x):
        gf = np.sum(matrix[i])
        df = np.count_nonzero(matrix[i])
        g[i] = gf/df
    return g

def global_ldf_weighting( matrix ):
    # for each word, square the values then sum them
    # return the inverse of that 
    (x,_) = matrix.shape
    g = np.zeros(x)

    for i in range(x):
        df = np.count_nonzero(matrix[i])
        g[i] = np.log2(( x /( 1 + df )))                       
    return g

def global_entropy_weighting( matrix ):
    # for each word, square the values then sum themi]
    # return the inverse of that 
    (x,_) = matrix.shape
    g = np.zeros(x)
    
    for i in range(x):
        temp = matrix[i]               
        gf = np.sum(matrix[i])
        temp = temp/gf
        a = temp
        for j in range(len(temp)):
            if temp[j] == 0:
                a[j] = 0
            else:
                a[j] = np.log2( temp[j] )
        temp = temp * a        
        temp = temp / np.log2( x )
        g[i] = 1 + np.sum( temp )    
    return g

def best_weighting( matrix ):
    g = global_entropy_weighting( matrix )
    A = matrix
    (rows,_) = A.shape
    for i in range(rows):
        A[i] = g[i] * np.log2(A[i]+1)
    return A

def weight_matrix(matrix, local_weight, global_weight):
    A = matrix
    B = matrix
    
    if local_weight == 'log' or local_weight == 1:
        A = local_log_weighting( A )
    elif local_weight =='aug' or local_weight == 2:
        A = local_aug_weighting( A )
    elif local_weight =='binary' or local_weight == 3:
        A = local_binary_weighting( A ) 
    else:
        A = A

    if global_weight == 'norm' or global_weight == 1:
        g = global_normal_weighting( B ) 
    elif global_weight == 'gfldf' or global_weight == 2:
        g = global_gfldf_weighting( B ) 
    elif global_weight == 'ldf' or global_weight == 3:
        g = global_ldf_weighting( B )
    elif global_weight == 'entropy' or global_weight == 4:
        g = global_entropy_weighting( B )
    else:
        g = np.array(1)
  
    return np.multiply(A.T, g).T

#-------------------------
# START HERE
#-------------------------

def JITT( freq, g, rank, filename, quizname ):
    # read the answer log given by the file name
    # choose the proper quiz name specified 
    # essays are always question 1, so manually specified
    answermatrix = read_answer_log( filename, quizname, [1])
    #print_to_file('rawfile.txt', answermatrix)

    #---------------------------------------------------
    # BIG DOC block
    #---------------------------------------------------

    # handle the pre-processing
    fixed_matrix = clean_up_data(answermatrix)
#    print np.array(fixed_matrix)
    # print to file
    print_to_file('BigDoc.txt', fixed_matrix)
    # construct the initial dictionary
    dictionary = count_all_words(fixed_matrix)    

    #---------------------------------------------------
    # DOC Matrix block
    #---------------------------------------------------

    # get the columns needed for the doc matrix    
    columns = set_dict_values_to_zero(dictionary)
    
    # construct doc matrix
    A_doc_matrix = word_matrix(fixed_matrix, columns)

    #---------------------------------------------------
    # Matrix Preprocessing block
    #---------------------------------------------------

    # preprocess the doc matrix
    # A_prime has words as columns, students as rows
    A_prime = weight_matrix( A_doc_matrix, freq, g )
    print_to_file('A_prime.txt', A_prime)
    
    words = dict_to_array( dictionary )
    # recreate the original matrix from a frequency matrix
    word_array = words[0:,0]
    
    print_to_file('word_array.txt', word_array)
    
    word_freq_A_prime = [ round(np.sum(row)) for row in A_prime.T ]
    zip_freq_to_words = dict(zip(word_array, word_freq_A_prime))

    # create something that wordle understands
    A_prime_wordle = recreate_wordle_matrix(zip_freq_to_words)
    print_to_file('ProcessedMatrix_wordle.txt', A_prime_wordle)

    #---------------------------------------------------
    # Matrix SVD block
    #---------------------------------------------------

    # only get the words that occur > 1
    dict_threshold = set_minimum( zip_freq_to_words, 1 )
    # get the words that do occur > 1, set their occurence to 0
    columns_thresh = set_dict_values_to_zero(dict_threshold)
    # new word matrix
    A_prime_thresh = word_matrix(fixed_matrix, columns_thresh)

    # word array for words that occur > 1
    words_thresh = dict_to_array( dict_threshold )
    # recreate the original matrix from a frequency matrix
    word_array_thresh = words_thresh[0:,0]
    
    # get SVD of the A_prime transpose matrix
    # transpose because, we want the words to be the
    # rows and students as columns
    # it's also weighted
    U,S,VT = np.linalg.svd(A_prime_thresh.T, full_matrices=False)

    # U should have #-of-words rows

    # fix rank first
    (x,_) = A_prime_thresh.shape
    if rank > x:
        rank = x

    U_k = np.array(U[:,:rank])
    S_k = np.array(S[0:rank])
    V_k = np.array(VT[0:rank])
    left = np.dot(U_k, np.eye(rank)*S_k)
    
   # print S_k
   # print V_k

    #----------------------------------------------
    #----------------------------------------------
    Puff = np.array([[ 0,  0,  1,  1,  0,  0,  0,  0,  0,],
    [ 0,  0,  0,  0,  0,  1,  0,  0,  1,],
    [ 0,  1,  0,  0,  0,  0,  0,  1,  0,],
    [ 0,  0,  0,  0,  0,  0,  1,  0,  1,],
    [ 1,  0,  0,  0,  0,  1,  0,  0,  0,],
    [ 1,  1,  1,  1,  1,  1,  1,  1,  1,],
    [ 1,  0,  1,  0,  0,  0,  0,  0,  0,],
    [ 0,  0,  0,  0,  0,  0,  1,  0,  1,],
    [ 0,  0,  0,  0,  0,  2,  0,  0,  1,],
    [ 1,  0,  1,  0,  0,  0,  0,  1,  0,],
    [ 0,  0,  0,  1,  1,  0,  0,  0,  0,]])
    
    Us,Ss,VTs = np.linalg.svd(Puff, full_matrices=False)

    # fix rank first
    (x,_) = A_prime_thresh.shape
    if rank > x:
        rank = x

    U_s = np.array(Us[:,:3])
    S_s = np.array(Ss[0:3])
    V_s = np.array(VTs[0:3])
    lol = np.dot(Us, np.eye(9)*Ss)
    lool= np.dot(lol,VTs)
    
    #print np.dot(np.dot(Us, Ss),VTs)
    #print U_s
    #print Ss
    #print VTs
    #print lool[0]
    #print Puff[0]
    
    Rank_k = np.dot( left, V_k )
    word_freq_Rank_k = [ round(np.sum(row)) for row in Rank_k ]
    Rank_k_dict = dict(zip(word_array_thresh, word_freq_Rank_k))
    Rank_k_wordle = recreate_wordle_matrix(Rank_k_dict)

    filename = 'Rank_' + str(rank) + '_wordle.txt'
    
    print_to_file(filename, Rank_k_wordle)

    top30 = np.array(sort_by_value(Rank_k_dict))[0:31]

    top30words = top30[:,0]
    wordset = set(top30words)

    
    #----------------------------------------------
    # this is without threshold ( all words, even ones that occured once )
    #----------------------------------------------

    print A_prime.shape
    Ul,Sl,VTl = np.linalg.svd(A_prime.T, full_matrices=False)

    # fix rank first
    (x,_) = A_prime_thresh.shape
    if rank > x:
        rank = x

    #print Sl

    ###--------------------------------------------

    # PLOTTING
    #-------------------------------
    
    U_plot = np.array(U[:,1:3])*1000
    V_plot = np.array(VT[1:3])
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    x_axis = np.array(word_array_thresh)
    p, = ax.plot(U_plot.T[0], U_plot.T[1], '.')
    row_anno = 0

    U_anno = U_plot
    for word in word_array_thresh:
        ax.annotate(word, (U_anno[row_anno][0],U_anno[row_anno][1]), size='xx-small')
        row_anno += 1

    temp = 0

    lol =  np.array(zip(range(125),np.array(word_array_thresh)))
    print_to_file('label.txt', lol)
   # plt.xticks(range(-100,100,1))#, x_axis, size='small', rotation=90)
    #ax.set_xlim(-50,50)
    #ax.set_ylim(-75,75) 
    #plt.show()
    plt.close()
    
    # GENSIM HERE ------------------
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)    

    texts = fixed_matrix
    
    dictionary = corpora.Dictionary(texts)
    mm = [dictionary.doc2bow(text) for text in texts]
   # for text in texts:
    #    print text
    # save
    #corpora.MmCorpus.serialize('Results/deerwester.mm', mm)
    
    # LSI
    lsi = gensim.models.lsimodel.LsiModel(corpus=mm, id2word=dictionary, num_topics=400)
    #lsi.print_topics(10)

    #print np.array(zip(lsi.projection.u[1],  Ul[1]))
   # print l
    print Ul.shape
    print lsi.projection.u.shape   
    #print np.sort(lsi.projection.u.T[0])
    #print lsi[mm]
    #print Sl.shape

    print lsi.projection.u.T[0,:]

    loy = np.array([[1,2],[3,4]])
    print loy.T[0,:]
    
    print lsi.projection.u.T
    #lol  = np.array(mm)
    #print lol.shape
    #print lol[0]
    #print np.array(mm).shape
    #print np.array(fixed_matrix).shape
    #print mm
    #print mm[1]
    #print mm[5]
    #print A_prime.T[0]
    #print fixed_matrix[5]
    #print dictionary.keys()

    
    # LDA
    #lda = gensim.models.ldamodel.LdaModel(corpus=mm, id2word=dictionary, num_topics=100, update_every=1, chunksize=10000, passes=5)
    #lda.print_topics(10)

    # END #-----------------------------------------------------------

    
if __name__=="__main__":
    lmtzr = WordNetLemmatizer()
    #print lmtzr.lemmatize('was')
    # freq, g, rank, log, quiz name 
    JITT( 0, 0, 3, 'answer_log', 'Lecture_6')
    #print if_time_stamp( 'lol' )
    #read_answer_log('answer_log', 'Lecture_1', [1])
   
