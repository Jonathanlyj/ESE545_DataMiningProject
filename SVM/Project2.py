import numpy as np
from scipy import sparse
import string
from sklearn.feature_extraction.text import CountVectorizer
import codecs
import random
import math
import matplotlib.pyplot as plt

def main():
    #problem 1 and 2
    train_data = read_in('training.1600000.processed.noemoticon.csv', 'stopwords.txt')
    processed_train_data = data_process(train_data)
    #problem 3
    bag_of_words=get_BoW(processed_train_data)
    extracted_train_data = unigram_features_extraction(processed_train_data,bag_of_words)
    #problem 4 and 5 
    result_Pegasos = PEGASOS_SVM(extracted_train_data,40001)
    result_Adagrad = AdaGrad(extracted_train_data,40001)
    write_results_to_file('Results.csv', result_Pegasos[1], result_Adagrad[1])
    #problem 6

    
    result_Pegasos = PEGASOS_SVM_1(extracted_train_data,40001)
    result_Adagrad = AdaGrad_1(extracted_train_data,40001)
    write_results_to_file('Results_test.csv', result_Pegasos[1], result_Adagrad[1])

def read_in(filename1,filename2):
    # read in sentiment and tweet infomation
    f1 = codecs.open(filename1, "r", encoding="latin-1")
    sentiment=[]
    tweet=[]
    for line in f1:
        line_info=(line.rstrip()).split(',')
        if len(line_info)>=5:
            sentiment.append(line_info[0].strip('"'))
            tweet.append(line_info[5].strip('"'))
        
    #read in stopwords info into a list
    f2=open(filename2)
    stopwords=[]
    next(f2)
    for line in f2:
        line_info=line.strip('\n')
        stopwords.append(line_info)
    return (sentiment,tweet,stopwords)


def data_process(data):
    (sentiment,tweet,stopwords) = data
    #convert 0 and 4 to -1 and 1 respectively
    sentiment=(np.array(sentiment)).astype(np.float)
    sentiment=(sentiment-2)/2

    stopwords.append(' ')

    #process tweet
    #convert all letters into lowercase.
    tweet= list(np.char.lower(np.array(tweet)))
  
    for i in range(len(tweet)):
        tweet_list=tweet[i].split(' ')
        
        #Conver all occuraces of web address to URL
        for j in range(len(tweet_list)):
            if 'www.' in tweet_list[j] or 'http' in tweet_list[j]:
                tweet_list[j]='URL'
            if tweet_list[j].startswith('@'):
                tweet_list[j]='AT-USER'

        #Replace duplicated words and eliminate stopwords, including additional whitespaces
        new_tweet_list=tweet_list
        tweet_list=[' ']
        for k in new_tweet_list:
            if k != tweet_list[-1]:
                if k not in stopwords:
                    tweet_list.append(k)

        tweet[i]=(' '.join(map(str,tweet_list))).strip()

        #Remove all punctuation
        for char in tweet[i]:
            if char in string.punctuation:
                tweet[i]=tweet[i].replace(char,'')

    return (sentiment,tweet,)

def get_BoW(processed_data):
    (sentiment,tweet) = processed_data
    new_tweet = []
    #convert words separated by space in strings into lists of words
    for item in tweet:
        new_item = item.split(' ')
        new_tweet.append(new_item)
    #obtain bag of word as the union of sets of all tweet lists
    bag_of_words=list(set().union(*new_tweet))
    
    return bag_of_words
    
    
def unigram_features_extraction(processed_data,bag_of_words):
    (sentiment,tweet) = processed_data
    

    #generate the featre list
    cv=CountVectorizer(vocabulary=bag_of_words)
    feature_list=cv.fit_transform(tweet)

    extracted_data = (feature_list,sentiment)
    return extracted_data


def error_rate_calcu(extracted_data,number_of_data,w):
    (feature_list,sentiment) = extracted_data
    #randomly pick 1000 samples to test the error rate for the classifier in this iteration
    error = 0
    index_list = random.sample(range(feature_list.shape[0]),number_of_data)
    for k in index_list:
        x = feature_list[k]
        y = sentiment[k]
        indexes = x.nonzero()[1]
        product = 0
        for j in indexes:
            product += x[0,j]*w[0,j]
        if y*product < 0:
            error += 1
    return error/number_of_data


def PEGASOS_SVM(extracted_data,max_iteration):
    (feature_list,sentiment) = extracted_data   
    lembda = 0.001 #regularization parameter
    #initialize the classifier w
    w = sparse.csr_matrix((1,feature_list.shape[1]))
    #determine the total iteration number and record the error rate for every 1000 iterations
    error_rates = []
    T = 0
    while T+1 <= max_iteration:
        for t in range(T+1,T+1000):
            #mini batch
            index_list = random.sample(range(feature_list.shape[0]),100)
            #mini_batch_positive
            products = 0
            for i in index_list:
                x = feature_list[i]
                y = sentiment[i]
                indexes = x.nonzero()[1]
                product = 0
                for j in indexes:
                    product += x[0,j]*w[0,j]
                if y*product < 1:
                    products += y*x
            #compute the gradient            
            gradient = lembda*w - (1/(t*lembda))/100*products
            w_ = w - (1/(t*lembda))*gradient
            w = min([1,1/math.sqrt(lembda)/sparse.linalg.norm(w_)])*w_

        error_rate = error_rate_calcu(extracted_data,1000,w)
        #problem 6
        #error_rate = error_rate_calcu(extracted_data_test,length,w)
        error_rates.append(error_rate) 
        T += 1000

    return (w,error_rates)

def AdaGrad(extracted_data,max_iteration):
    (feature_list,sentiment) = extracted_data

    #regularization parameter
    lembda = 0.001 
    #master_stepsize
    η = 1 
    #fudge_factor for numerical stability
    fudge_factor = 1e-6
    w = sparse.csr_matrix((1,feature_list.shape[1]))
    #determine the total iteration number and record the error rate for every 1000 iterations
    error_rates = []
    T = 0
    historical_grad = 0
    while T+1 <= max_iteration:
        for t in range(T+1,T+1000):
            #same as the part in pegasos
            index_list = random.sample(range(feature_list.shape[0]),100)
            products = 0
            for i in index_list:
                x = feature_list[i]
                y = sentiment[i]
                indexes = x.nonzero()[1]
                product = 0
                for j in indexes:
                    product += x[0,j]*w[0,j]
                if y*product < 1:
                    products += y*x

            gradient = (lembda*w - (1/(t*lembda))/100*products).toarray()
            #adjust the gradient in this iteration
            historical_grad += gradient**2
            adjusted_grad = sparse.csr_matrix(gradient / (fudge_factor + np.sqrt(historical_grad)))
            #update the classifier w
            w = w - η*adjusted_grad

        error_rate = error_rate_calcu(extracted_data,1000,w)
        #problem 6
        #error_rate = error_rate_calcu(extracted_data_test,length,w)
        error_rates.append(error_rate) 
        T += 1000

    return(w,error_rates)

def write_results_to_file(filename, result_Pegasos, result_Adagrad):
    #write the error rate for every 1000 iterations of Pegasos and result_Adagrad 
    #into the first and second column of the result file respectively 
    outfile = open(filename, 'w')
    for i in range(len(result_Pegasos)):
        to_write = str(result_Pegasos[i])+','
        to_write += str(result_Adagrad[i])+'\n'
        outfile.write(to_write)

#problem 6
def test_data_(filename1,filename2,filename3):
    
    train_data = read_in(filename1,filename3)
    test_data = read_in(filename2,filename3)
    for i in range(len(test_data[0])):
        if test_data[0][i] == '2':
            test_data[0][i]='4'
    
    processed_train_data = data_process(train_data)
    processed_test_data = data_process(test_data)
    #Bag of words still need to be generated from train data
    bag_of_words=get_BoW(processed_train_data)
    extracted_test_data = unigram_features_extraction(processed_test_data,bag_of_words)
    
    return (extracted_test_data,len(test_data[0]))

def PEGASOS_SVM_1(extracted_data,max_iteration):
    (feature_list,sentiment) = extracted_data
    (extracted_data_test,length) = test_data_('training.1600000.processed.noemoticon.csv','testdata.manual.2009.06.14.csv','stopwords.txt')    
    lembda = 0.001 #regularization parameter
    #initialize the classifier w
    w = sparse.csr_matrix((1,feature_list.shape[1]))
    #determine the total iteration number and record the error rate for every 1000 iterations
    error_rates = []
    T = 0
    while T+1 <= max_iteration:
        for t in range(T+1,T+1000):
            #mini batch
            index_list = random.sample(range(feature_list.shape[0]),100)
            #mini_batch_positive
            products = 0
            for i in index_list:
                x = feature_list[i]
                y = sentiment[i]
                indexes = x.nonzero()[1]
                product = 0
                for j in indexes:
                    product += x[0,j]*w[0,j]
                if y*product < 1:
                    products += y*x
            #compute the gradient            
            gradient = lembda*w - (1/(t*lembda))/100*products
            w_ = w - (1/(t*lembda))*gradient
            w = min([1,1/math.sqrt(lembda)/sparse.linalg.norm(w_)])*w_

        error_rate = error_rate_calcu(extracted_data_test,length,w)
        error_rates.append(error_rate) 
        T += 1000

    return (w,error_rates)

def AdaGrad_1(extracted_data,max_iteration):
    (feature_list,sentiment) = extracted_data
    (extracted_data_test,length) = test_data_('training.1600000.processed.noemoticon.csv','testdata.manual.2009.06.14.csv','stopwords.txt')
    #regularization parameter
    lembda = 0.001 
    #master_stepsize
    η = 1
    #fudge_factor for numerical stability
    fudge_factor = 1e-6
    w = sparse.csr_matrix((1,feature_list.shape[1]))
    #determine the total iteration number and record the error rate for every 1000 iterations
    error_rates = []
    T = 0
    historical_grad = 0
    while T+1 <= max_iteration:
        for t in range(T+1,T+1000):
            #same as the part in pegasos
            index_list = random.sample(range(feature_list.shape[0]),100)
            products = 0
            for i in index_list:
                x = feature_list[i]
                y = sentiment[i]
                indexes = x.nonzero()[1]
                product = 0
                for j in indexes:
                    product += x[0,j]*w[0,j]
                if y*product < 1:
                    products += y*x

            gradient = (lembda*w - (1/(t*lembda))/100*products).toarray()
            #adjust the gradient in this iteration
            historical_grad += gradient**2
            adjusted_grad = sparse.csr_matrix(gradient / (fudge_factor + np.sqrt(historical_grad)))
            #update the classifier w
            w = w - η*adjusted_grad

        error_rate = error_rate_calcu(extracted_data_test,length,w)
        error_rates.append(error_rate) 
        T += 1000

    return(w,error_rates)

main()
##print(test_data('testdata.manual.2009.06.14.csv','stopwords.txt')[0][0].toarray()[0][0])

   
