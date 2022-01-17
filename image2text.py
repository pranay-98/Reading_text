#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (Parth Kapil, Harishanker Brahma Kande, Pranay Reddy Dasari)
# (based on skeleton code by D. Crandall, Oct 2020)
#

from PIL import Image, ImageDraw, ImageFont
import sys
from collections import defaultdict
import math
import numpy as np

CHARACTER_WIDTH=14
CHARACTER_HEIGHT=25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    # print(im.size)
    # print(int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH)
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [ [ "".join([ '*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg+CHARACTER_WIDTH) ]) for y in range(0, CHARACTER_HEIGHT) ], ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return { TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS) ) }

#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)

## Below is just some sample code to show you how the functions above work. 
# You can delete this and put your own code here!

#class for recognizing letters in the image
class DetectLetters:
    
    #in constructor we just initialize the default values
    def __init__(self):
        self.initial_probs = defaultdict(int)
        self.transition_probs = defaultdict(lambda: defaultdict(int))
        self.emission_probs = defaultdict(lambda: defaultdict(int))
        
        self.train_letters = None
        self.test_letters = None
        
    
    #read the data from a given file name
    def read_file_data(self, file_name):
        file  = open(file_name, mode="r")        
        file_content = file.read()        
        file.close()
        
        return file_content    
    
    #removes all the char which are not in the valid char set
    def clean_file(self, content):
        
        #As mentioned in the assignment we are assuming that the valid chars will be english letters only
        valid_chars = set([char for char in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "])
        
        #variable for final filtered string
        filtered_content = ''        
        for char in content:
            if char not in valid_chars:
                continue
            
            filtered_content += char
            
        return filtered_content
    
    #calculates the probability of each character with which a sentence start in the given train file
    def get_initial_probs(self, train_file):
        initial_probs = defaultdict(int)
        
        #variable to check if current character is the first letter of the sentence
        first_char = True
        
        #looping through each char in file
        for char in train_file: 
            
            #if the char is first letter and char is not space we increament the value of the character   
            if first_char and char != " ":            
                initial_probs[char] += 1
             
            # if current char is period then the next char will be the first letter of the next sentence   
            if char == ".":
                first_char = True
          
          
        #getting the total sum of all the value counts      
        val_sum = sum(initial_probs.values())
        
        #calculating log probabilty for each char
        for char, value in initial_probs.items():
            initial_probs[char] = np.log(value/val_sum)
        
        
        return initial_probs
        
       
    #calculate transition probabilities
    def get_transition_probs(self, train_file):
        trans_probs = defaultdict(lambda: defaultdict(int))
        
        #get count for transition form one character to another
        for ind in range(0, len(train_file)-2):
            curr_char, next_char = train_file[ind], train_file[ind+1]   
            
            trans_probs[curr_char][next_char] += 1
        
        #calculating the log probability for each char
        for prob_dict in trans_probs.values(): 
            #sum of all the values for this char       
            val_sum = sum(prob_dict.values())
            
            #calculating probabilty for each char
            for char, value in prob_dict.items():
                prob_dict[char] = np.log(value/val_sum)
            
        return trans_probs 
    
    
    #get probability of matching pixels between teat and train patterns
    def get_maching_prob(self, tst_pattern,train_pattern):
        sim_prob=1
        
        #probability of 1 for when both pixels are star
        valid_star_prob=1         
        #probability of 0.5 for when both pixels are space
        valid_space_prob=0.5       
        #probability of 0.1 for when there is no match with both pixels
        invalid_prob=0.1
        
        for row in range(len(tst_pattern)):
            for col in range(len(tst_pattern[0])):
                #if pixels don't match
                if tst_pattern[row][col]!=train_pattern[row][col]:
                    sim_prob *= invalid_prob
                else:
                    if (tst_pattern[row][col]=='*'):
                        sim_prob *= valid_star_prob
                    else:
                        sim_prob *= valid_space_prob
        return sim_prob     

    #calculate emission probability for each character in the test_letters with all the 
    #characters in the train_letters 
    def get_emission_probs(self):
        emission_probs = defaultdict(lambda: defaultdict(int))

        for tst_ind, test_pattern in enumerate(self.test_letters):
            for train_char, train_pattern in self.train_letters.items():
                prob = self.get_maching_prob(test_pattern, train_pattern)
                emission_probs[tst_ind][train_char] = np.log(prob)  
                
        return emission_probs    
        
    #reads the train file and calculates the initial, transition, emission probs
    def train(self, train_letters, train_file_name,  test_letters):
        print("Training model ............")
        
        
        #setting train and test letters in the object
        self.train_letters = train_letters
        self.test_letters = test_letters        
        
        #read file
        train_file = self.read_file_data(train_file_name)
        
        #clean_file
        train_file = self.clean_file(train_file)
        
        #get initial probs
        self.initial_probs = self.get_initial_probs(train_file)
        
        #get transition probs
        self.transition_probs = self.get_transition_probs(train_file)
        
        #get emission probs
        self.emission_probs = self.get_emission_probs()
        
        print("Training Done!")


    #returns the result for the simple bayes net
    def getBayesNetResult(self):
        print("Simple: ", end="")
        
        #for each char in the testing data we are printing the corresponding train character which has the maximum 
        # emission probability
        for ind in range(len(self.emission_probs.keys())):
            print(max(self.emission_probs[ind], key=self.emission_probs[ind].get), end="")
       
    #runs the viterbi algorithm and returns the most probable sequence of results
    def getViterbiResult(self):
        #initailizing the viterbi matrix with rows = number of training chars and columns = number of testing chars
        viterbi_mat = [[None for _ in range(len(self.test_letters)) ] for i in range(len(self.train_letters.keys()))]
        
        #calculating probabilities for the first char in the testing chars
        for ind, char in enumerate(self.train_letters.keys()):
            viterbi_mat[ind][0] = (-1, self.emission_probs[0][char] + self.initial_probs[char])

        # Calculating probabilities for each letter in the testing chars apart from the first char
        for test_ind in range(1, len(self.test_letters)):
            for train_ind, curr_char in enumerate(self.train_letters.keys()):                
                best_char_prob = (-1, float("-inf"))
                
                for prev_train_ind, prev_char in enumerate(self.train_letters.keys()):
                    
                    #calculating prob for current train char and the current test char
                    curr_prob = viterbi_mat[prev_train_ind][test_ind - 1][1] + self.transition_probs[prev_char][curr_char] + self.emission_probs[test_ind][curr_char]
                    
                    #if current train char prob is greater than the previous best train char we change the values
                    if curr_prob > best_char_prob[1]:
                        best_char_prob = (prev_train_ind, curr_prob)
                        
                #storing the best values in the viterbi matrix
                viterbi_mat[train_ind][test_ind] = (best_char_prob[0], best_char_prob[1])
        
        #backtracking to get the best sequence of letters
        (best_ind, best_prob) = (-1, float("-inf"))
        
        #getting the best train char prob from the last column of the viterbi matrix
        for ind in range(len(self.train_letters.keys())):
            char_prob = (viterbi_mat[ind][len(self.test_letters) - 1][1])
            if char_prob > best_prob:
                (best_ind, best_prob) = (ind, char_prob)

        #list for storing the best sequence of chars
        result_seq = list()
        
        #list of all the training chars
        train_char_list = list(self.train_letters.keys())
        
        # inserting th best hcar in the result_seq list
        for col in range(len(self.test_letters) - 1, 0, -1):
            result_seq.insert(0, train_char_list[best_ind])
            best_ind = viterbi_mat[best_ind][col][0]
        result_seq.insert(0, train_char_list[best_ind])
        print ('HMM:', ''.join(result_seq))
            


#creating object of our custom class
model = DetectLetters()

#training the model on the given txt file and given patterns for train and test characters(chars)
model.train(train_letters, train_txt_fname, test_letters)

#Getting the result for the simple bayes net
model.getBayesNetResult()
print()

#getting the result for the viterbi algorithm
model.getViterbiResult()


