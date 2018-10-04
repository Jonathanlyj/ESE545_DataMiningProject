import scipy
import random
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import codecs



def main():
  ratings = read_in_and_process("ratings.dat","users.dat","movies.dat")
  (F_record1,Time_record1)=greedy(50, ratings)
  (F_record2,Time_record2)=lazy_greedy(50, ratings)
  x_list = list(range(10,51,10))
  
  #plot the max F values versus k using Greedy Algorithm
  plt.figure(1)
  plt.plot(x_list,F_record1,'-o')
  plt.title('F(A) versus k using Greedy Algorithm' )
  plt.xlabel(u'k')
  plt.ylabel('F(A)')
  plt.show()

  #plot the max F values versus k using Lazy Greedy Algorithm
  plt.figure(2)
  plt.plot(x_list,F_record2,'-o')
  plt.title('F(A) versus k using Lazy Greedy Algorithm' )
  plt.xlabel(u'k')
  plt.ylabel('F(A)')
  plt.show()

  #plot the runtime versus k under two different algorithm
  plt.figure(3)
  plt.plot(x_list,Time_record1,'-o',label=u'Greedy Algorithm')
  plt.plot(x_list,Time_record2,'-o',label=u'Lazy Greedy Algorithm')
  plt.legend()
  plt.title('Runing time under different algorithm' )
  plt.xlabel(u'k')
  plt.ylabel('Time/seconds')
  plt.show()

#read in movie info, user info, ratings info respectively into three three collections
def read_in_and_process(filename1,filename2,filename3):
  stream1 = codecs.open(filename1, "r", encoding="latin-1")
  stream2 = codecs.open(filename2, "r", encoding="latin-1")
  stream3 = codecs.open(filename3, "r", encoding="latin-1") 

  #read in movie info into a dictionary
  i = 0
  movie_info = {}
  movie_index = {}
  for line in stream3:
    line_info = line.split("::")
    movie_info[int(line_info[0])] = line_info[1] + line_info[2]
    movie_index[int(line_info[0])] = i 
    i = i + 1

  #read in user info into a dictionary
  user_info = {}
  for line in stream2:
    line_info = line.split("::")
    user_info[int(line_info[0])] = line_info[1:]

  #read in ratings info into a matrix
  ratings = np.zeros((len(user_info),len(movie_info)))
  for line in stream1:
    line_info = line.split("::")
    ratings[int(line_info[0])-1][movie_index[int(line_info[1])]-1] = line_info[2]

  return ratings

#greedy submodular maximization algorithm
def greedy(k,ratings):
  A = np.zeros(len(ratings))
  ratings = np.transpose(ratings)
  F_record = []
  Time_record = []
  time0=time.time()
  #keep adding movie vectors into subset A and find the maximum F value
  for i in range(k):
    #determine the maximum value vector between A and the rating vector of each movie
    #and return an matrix representing max vectors
    max_matrix = np.maximum(A,ratings)
    #find the maximum F value over all vectors
    F_max = np.max(np.mean(max_matrix,axis=1))
    #put the rating vector in the ratings matrix that lead to the max F value into subset A
    #and return a new max vector A
    A = max_matrix[np.argmax(np.mean(max_matrix,axis=1))]

    #record the max F value and time consumed every 10 round
    if (i+1) % 10 == 0:
      F_record.append(F_max)
      time1=time.time()
      Time_record.append(time1-time0)

  return (F_record,Time_record)

def lazy_greedy(k,ratings):
  ratings = np.transpose(ratings)

  #initalization
  #calculate the original F vector
  mean_vector =  np.mean(ratings,axis=1)
  #determine the original subset A with the largest F value
  A = ratings[np.argmax(mean_vector)]
  #determine the original max F value
  F_max=np.max(mean_vector)
  #set the largest F value in the F vector to 0 so it is easy to find the second largest F value and its index 
  mean_vector[np.argmax(mean_vector)]=0
  #determine the initial delta vectorb by sorting F vector
  sorted_deltas = np.sort(mean_vector*(-1))*(-1)

  F_record = []
  Time_record = []
  time0=time.time()

  for i in range(1,k):
    #calculate delta2 in new round using the updated F vector and the last max F value
    delta = np.mean(np.maximum(ratings[np.argmax(mean_vector)],A))-F_max
    #comparing the delta 2 in this round with the delta 3 in the last round
    #if delta 2 is larger, because of submodularity, choose this movie vector
    #and put this vector into subset A
    if round(delta,2) >= round(sorted_deltas[1],2):
      #update max F value, subset A, mean_vector and delta vector
      F_max = delta + F_max
      A = np.maximum(ratings[np.argmax(mean_vector)],A)
      mean_vector[np.argmax(mean_vector)]=0
      sorted_deltas = np.delete(sorted_deltas,0)
    #if not, reorder the F_vector and choose the movie vector leading to the largest F value 
    else:
      #reorder with the current subset A 
      max_matrix = np.maximum(A,ratings)
      mean_vector = np.mean(max_matrix,axis=1)
      #update max F value, A, mean_vector and delta vector respectively
      A = np.maximum(max_matrix[np.argmax(mean_vector)],A)
      F_max_last = F_max
      F_max = np.max(mean_vector)
      mean_vector[np.argmax(mean_vector)]=0
      sorted_deltas = np.sort((mean_vector-F_max_last)*(-1))*(-1)
      
    #record the max F value and time consumed every 10 round
    if (i+1) % 10 == 0:
      F_record.append(F_max)
      time1=time.time()
      Time_record.append(time1-time0)

  return (F_record,Time_record)

main()









    
 





