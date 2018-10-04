import scipy
import random
import itertools
import collections
import numpy as np
import matplotlib.pyplot as plt


def main():
        # question_1
        file_info = read_in_and_process('ratings.csv')
        matrix = Creating_matrix(file_info)
        # question_2
        (average_sim, J_LAR) = similarity(matrix)
        print('The average Jaccard similarity of randomly picked 10000 user pairs is:',average_sim,'\n')
        print('The ten largest jaccard similarity among them is:', J_LAR)
        #question_3
        users_info = efficiency_data_storage(file_info) 
        #question_4
        signature_matrix = minhash(users_info,1000)                     
        similar_users_LSH = hash_to_buckets(signature_matrix,100,10)    
        very_similar_users = find_actual_simi_users(similar_users_LSH, users_info)
        write_results_to_file('very_similar_users.csv', very_similar_users)
        #question_5
        actual_similar_user = neighbor_query_point(7,signature_matrix,users_info)
        print(actual_similar_user)


def quetion_1():
        file_info = read_in_and_process('ratings.csv')
        matrix = Creating_matrix(file_info)
        return matrix

def question_2():
        (average_sim, J_LAR) = similarity(quetion_1)
        print('The average Jaccard similarity of randomly picked 10000 user pairs is:',average_sim,'\n')
        print('The ten largest jaccard similarity among them is:', J_LAR)

def question_3():
        file_info = read_in_and_process('ratings.csv')
        users_info = efficiency_data_storage(file_info)
        return users_info

def question_4():
        file_info = read_in_and_process('ratings.csv')
        users_info = efficiency_data_storage(file_info)
        signature_matrix = minhash(users_info,1000)
        similar_users_LSH = hash_to_buckets(signature_matrix)
        very_similar_users = find_actual_simi_users(similar_users_LSH, users_info)
        write_results_to_file('1.csv', very_similar_users)
        return very_similar_users

def question_5(user_ID):
        file_info = read_in_and_process('ratings.csv')
        users_info = efficiency_data_storage(file_info)
        signature_matrix = minhash(users_info,1000)
        actual_similar_user = neighbor_query_point(user_ID,signature_matrix,users_info)
        return actual_similar_user


def read_in_and_process(filename):
        #read in info into three lists
        f = open(filename)  
        next(f)
        userids = [] 
        movieids = []
        ratings = []
        for line in f:
                line_info = (line.rstrip()).split(',')
                userids.append(line_info[0])
                movieids.append(line_info[1])
                ratings.append(line_info[2])

        #simplify the ratings greater than 2.5 to True, less and euqal to 2.5 to False
        ratings = (np.array(ratings)).astype(np.float)
        new_ratings = 2.5 < ratings

        #match each movieid with its row number by dictionary
        movieids_sorted = sorted(np.unique(movieids),key=int)
        succ_num = {}
        j = 0
        for i in movieids_sorted:
                succ_num[i] = j
                j = j + 1

        return ((userids,movieids,new_ratings,succ_num))


#creat a M*N rating matrix
def Creating_matrix(file_info):
        (userids,movieids,ratings,succ_num) = file_info
        #Create an all-zero matrix with the size M*N(movie_number*user_number)
        user_number = len(set(userids))
        movie_number = len(set(movieids))
        matrix = np.zeros((movie_number,user_number), dtype = np.single)
        #then change the correspondding zero to one according to the ratings 
        for k in range(0,len(ratings)):
                if ratings[k]:
                        movieid = movieids[k]
                        userid = userids[k]
                        movieid_num = succ_num[movieid]
                        matrix[movieid_num][int(userid)-1] = 1
        return matrix

# calculate Jaccard Similarity of randomly picked 10000 user pairs
def similarity(matrix):
        #randomly select distinct 10000 samples of user pairs
	total_users = list(range(0,len(matrix[0])))
	user_pairs = []      
	while len(user_pairs) < 10000:
		user_pair = random.sample(total_users,2) 
		if user_pair not in user_pairs:
			user_pairs.append(user_pair)

        #calculate the Jaccard similarity for each pair and add them into a list 
	J_sims = []
	J_LAR = []
	for pair in user_pairs:
                #use dot multiplication of vector to calculation the intersection of two users in a pair
		intersection = np.dot(matrix[:,pair[0]],matrix[:,pair[1]])
                #use the number of nonzero value in the sum of two vectors to calculate the union of two users in a pair
		union = np.size(np.nonzero(matrix[:,pair[0]]+matrix[:,pair[1]]) ) 
		J_sim = intersection/union
		J_sims.append(J_sim)

	#calculate the average value of the entire similarity list
	average_sim = np.mean(J_sims)
        #sort the list to obtain 10 largest similarities
	J_sims.sort(key = float)
	J_LAR = J_sims[-10:]

	#plot the histogram
	plt.hist(J_sims,bins=100)
	plt.title('Jaccard similarity over 10000 pairs of users')
	plt.xlabel('similarity')
	plt.ylabel('user pairs')
	plt.show()

	return((average_sim,J_LAR))


def efficiency_data_storage(file_info):
        (userids,movieids,ratings,succ_num) = file_info
        #only store successive numbers of the movies a user likes into the user list
        user_number = len(set(userids))
        movie_number = len(set(movieids))
        users_info = []
        for i in range(0,user_number):
                users_info.append([])
        for j in range(0,len(ratings)):
                if ratings[j]:
                        movieid = movieids[j]
                        userid = userids[j]
                        movieid_num = succ_num[movieid]
                        users_info[int(userid)-1].append(movieid_num)            
        return users_info

# generate a k length list with the value randomly picked form 1 to a prime number
# used for the pickup of coefficient in the minhash function
def pickcoefficient(k):
        #P represents a prime number larger than the total number of movieid
        prime=26759
        randomlist = []
        while k > 0:
                rand_num = random.randint(1, prime)
                while rand_num in randomlist:
                        rand_num = random.randint(1, prime)
                randomlist.append(rand_num)
                k = k - 1
        return randomlist

def minhash(users_info,hash_num):
        #a prime number larger than the total number of movieids
        prime=26759
        #creat two arrays where the values of coefficient in minhash function take from
        coefficientA = np.array(pickcoefficient(hash_num)).reshape(hash_num,1)
        coefficientB = np.array(pickcoefficient(hash_num)).reshape(hash_num,1)
        signature_matrix = []
        zero_vector = list((np.zeros((1,1000))+prime).flatten())
        for user in users_info:
                #use calculation of matrix(numpy) to calculate hash_num minhash values for each user as a signature vector
                hash_matrix = (coefficientA*(np.array(user))+coefficientB) % prime
                #then add these signature vectors to a signature matrix in which the columns represent user and rows represent minhash values 
                #with respect to those users who do not like any movies, represent all its minhash value as the largest prime number
                if hash_matrix.size != 0:
                        new_hash_matrix = (hash_matrix.min(1)).tolist()
                        signature_matrix.append(new_hash_matrix)                     
                else:
                        signature_matrix.append(zero_vector)
        return signature_matrix

def hash_to_buckets(signature_matrix,b,r):  
        #a prime number larger than the total number of movieids
        prime=26759
        #since the row of the signature matrix we get from minhash represents userid, we should transpose it first
        signature_matrix_1 = np.transpose(signature_matrix)
        similar_users = []
        #create two arrays where the values of coefficient in hash function take from
        coefficientA = np.array(pickcoefficient(1000)).reshape(1000,1)
        coefficientB = np.array(pickcoefficient(1000)).reshape(1000,1)
        #compute the result of hvi function for each user using vector multiplication
        #each row user different coefficients
        signature_matrix_2 = (coefficientA*(np.array(signature_matrix_1))+coefficientB) % prime
        #divide the hashed signature matrix into b bands and r rows
        for k in range(0,b):
                band = signature_matrix_2[r*k:r*k+r]
                new_band = band.sum(0)
                #if the results of multiple users are equal then hash the combination into one bucket
                dups = collections.defaultdict(list)
                for index,item in enumerate(new_band):
                        dups[item].append(index)
                for key in dups:
                        if len(dups[key]) > 1 :
                                similar_users.append(dups[key])

        return similar_users

#Given lists in which each value is unique, use intersection and union method to calculate the Jaccard similarity
def cal_Jacard_similarity(a,b):
        intersect = len(set(a).intersection(b))
        union = len(set(a).union(b))
        Jaccard_simi = intersect/union
        return Jaccard_simi

def find_actual_simi_users(similar_users, users_info):
        #since the buckets got from different bands could be identical, remove duplicates to reduce the amount of calculation
        similar_users.sort()
        similar_users = list(similar_users for similar_users,_ in itertools.groupby(similar_users))
        actual_simi = []
        for pairs in similar_users:
                # find out all non-repeated pairs in a bucket
                for pair in itertools.combinations(pairs,2):
                        # ignore the users who do not like any movies
                        if (len(users_info[pair[0]])!=0) and (len(users_info[pair[1]])!=0):
                                #determine whether the Jaccard similarity of a pair is above 0.65
                                if cal_Jacard_similarity(users_info[pair[0]], users_info[pair[1]]) >= 0.65:
                                        actual_simi.append((pair[0]+1, pair[1]+1))
        actual_simi.sort()
        actual_simi = list(actual_simi for actual_simi,_ in itertools.groupby(actual_simi))

        return actual_simi

# generate a csv file as a result
def write_results_to_file(filename, results):
        outfile = open(filename, 'w')
        for result in results:
                to_write = str(result[0])+','
                to_write += str(result[1])+'\n'
                outfile.write(to_write)


def neighbor_query_point(user_ID,signature_matrix,users_info):
        prime = 26759
        signature_matrix_1 = np.transpose(signature_matrix)
        '''Based on the signature matrix from 1000 hash function, find the nearest users with target user by repeatedly run LSH function with
           smaller and smaller threshold value until candidate neighbors are less than 10'''
        #compute the result of hvi function for each row using vector multiplication
        #each row uses different coefficients
        coefficientA = np.array(pickcoefficient(1000)).reshape(1000,1)
        coefficientB = np.array(pickcoefficient(1000)).reshape(1000,1)
        signature_matrix_2 = (coefficientA*(np.array(signature_matrix_1))+coefficientB) % prime
        hash_band = []


        candidate = {}
        actual = []
        probability = []
        h = 0
        stop = False
        while not stop:
                candidate[h]=[]
                #b*r is fixed to be 1000,with increment of h.update r and b so that threshold similarity increases
                r=3 + h
                b=1000//r
                #the probability is updated with new r and b
                probability.append((1/b)**(1/r))

                for k in range(0,b):
                        #pick slices as bands
                        band = signature_matrix_2[r*k:r*k+r]
                        new_band = band.sum(0)
                        for i in range(len(new_band)):
                                if i != (user_ID-1):
                                        if new_band[i]==new_band[user_ID-1]:
                                                candidate[h].append(i)
                candidate[h] = set(candidate[h])
                        
                if len(candidate[h]) < 10:
                        #stop the loop of LSH function if number of  candidate is reduced to less than 10
                        stop = True
                        similarities = []
                        userIDs = []
                        most_similar = []
                        #then select the similar neighbors from the current candidate group by calculating similarities
                        for user in candidate[h]:
                                similarity = cal_Jacard_similarity(users_info[user_ID-1], users_info[user])
                                if similarity >= probability[h]:
                                        similarities.append(similarity)
                                        userIDs.append(user+1)
                        '''if all candidate users are wrong(does not satisfy the current similarity),then access to
                        last candidate group(with last threshold value) and treat them as the similar users'''
                        if len(similarities) == 0:
                                for user in candidate[h-1]:
                                        similarity = cal_Jacard_similarity(users_info[user_ID-1], users_info[user])
                                        similarities.append(similarity)
                                        userIDs.append(user+1)
                        #examine each user in the similar users and find the most similar neighber(s)
                        for i in range(len(similarities)):
                                if similarities[i] == max(similarities):
                                        most_similar.append(userIDs[i])
                        
                h = h + 1
        return most_similar

main()



