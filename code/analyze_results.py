# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 13:51:22 2022

@author: bpe043
"""

import pandas as pd
import sqlite3
import os

def uncertainty_check(row):
    
    pattern = ['?', '@']
    
    if row['orig_manual'] is not None:
        for x in row['orig_manual']:
            if x in pattern:
                return True
    
    if row['manual'] is not None:
        for y in row['manual']:
            if y in pattern:
                return True
        
    return False


def get_number_of_self_copies(frame):
    
    users = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    results_dict = {}
    
    for u in users:
        temp = frame[frame['user'] == u]
        freqs = temp.orig_user.value_counts(normalize = True)
        results_dict[u] = freqs
        
    return results_dict


# Did the reviewers pay attention while transcribing?
# ~14% of each reviewer's copy images were from their own "original" images, this way we can see if they answered the same for both their original and their copy
# This can be used to measure if the reviewer were focused while working 
def attention_test(frame):
    
    users = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
    results_dict = {}
    numbers = []
    
    for u in users:
        temp = frame[frame['user'] == u]
        temp = temp[temp['orig_user'] == u]
        
        numbers.append(len(temp))
        
        # Remove rows where they answered "None" (meaning that they agreed with the ML label) for both images
        temp = temp[~(temp['manual'].isnull() & temp['orig_manual'].isnull())]
        
        # Get all rows where they answered differently for the original and the copy images
        temp = temp[temp['manual'] != temp['orig_manual']]
        
        # If there were no inconsistencies, return 0
        if len(temp) == 0:
            results_dict[u] = 0
        else:
            results_dict[u] = temp
            
    return results_dict, numbers
        
    


frame = pd.read_sql("select * from fields", con = sqlite3.connect("results.db"))
frame.columns = [x.lower() for x in frame.columns]


# Looking at the copy images, to see if we have gotten conflicting results or not
frame = frame[['name', 'code', 'manual', 'user']]

copy_frame = frame[frame['name'].str.endswith('copy.jpg')]
copy_frame['img_name'] = copy_frame['name'].str.split('copy', 1, expand = False).str[0] + '.jpg'
copy_frame[['orig_name', 'orig_manual', 'orig_user']] = None


# Frame with all the copies excluded
unique_frame = frame.drop(copy_frame.index)


copy_frame = copy_frame.reset_index(drop = True)
for index, row in copy_frame.iterrows():
    
    orig = frame[frame['name'] == row['img_name']]
    
    if len(orig) == 0:
        continue
    
    orig = orig[['name', 'manual', 'user']]
    orig = orig.reset_index(drop = True)
    
    copy_frame.at[index, 'orig_name'] = orig['name'].values[0]
    copy_frame.at[index, 'orig_manual'] = orig['manual'].values[0]
    copy_frame.at[index, 'orig_user'] = orig['user'].values[0]
    
    perc_done = (index/len(copy_frame)) * 100
    print("{:.2f}% done!".format(perc_done))
    
    

# Remove the images where a user had Both the original and copy image
self_copy_frame, self_copy_frame_numbers = attention_test(copy_frame)
copy_frame_valid = copy_frame[copy_frame.user != copy_frame.orig_user]  # New baseline for calculating percentages, 7749 instead of 9000

results_copy_frame = copy_frame_valid.copy()

# Both the manual labelers agreed with the ML label = 29.73% of images
# (When self-copies have been removed, we have = 29.24%)
agree_nones = results_copy_frame[(results_copy_frame.manual.isnull()) & (results_copy_frame.orig_manual.isnull())]

results_copy_frame = results_copy_frame.drop(agree_nones.index)

# Process the two columns to make sure the format is as similar as possible
results_copy_frame['manual'] = results_copy_frame['manual'].str.replace(' ', '')
results_copy_frame['orig_manual'] = results_copy_frame['orig_manual'].str.replace(' ', '')

# The two manual labelers agreed with each other, but Not the ML label = 83.5% (of remaining images [6324] after agree_nones have been dropped)
# (After self-copies... = 81.12% of remaining images [5483])
agree = results_copy_frame[results_copy_frame.manual == results_copy_frame.orig_manual]

# The two manual labelers disagreed with each other = 16.5%
# (After self-copies... = 18.87%)
disagree = results_copy_frame[results_copy_frame.manual != results_copy_frame.orig_manual]

# If the two columns contains a similar string even partially = 20.11%
# (After self-copies... = Here the comparison breaks down a bit, because I can't remember how I got 20.11%, the difference in pure images is only 2 images, and when doing
# the math again for copy images, we get 3.32% and then without copy images we get 3.8%. I will just continune the comparison for the main categories below)
partial_agree = pd.DataFrame()

for index, row in disagree.iterrows():
    
    if row['manual'] is not None:
        if row['orig_manual'] is not None:
            if (row['manual'] in row['orig_manual']) or (row['orig_manual'] in row['manual']):
                partial_agree = partial_agree.append(row, ignore_index = True)
                
                
                
# The manual labeler agreed with the ML label = 31.92%
manual_agreed_with_ml = unique_frame[unique_frame.manual.isnull()]

# Divide the copy-images into categories: (Numbers have been gotten from the code below)
# Certain -> Both manual labelers agreed with each other and with the ML label                 =  29.73% of images (9000) [agree_nones]
# Fairly_certain -> Both manual labelers agreed with each other, but not with the ML label     =  56.50% of images (9000) [agree]
# Unknown -> The manual labelers disagreed with each other (And maybe with the ML label)       =   6.50% of images (9000) []
# Uncertain -> The manual labelers used an "uncertainty" symbol                                =   7.26% of images (9000) 

certain = agree_nones.copy()

fairly_certain = results_copy_frame[results_copy_frame.manual == results_copy_frame.orig_manual]

unknown = results_copy_frame[results_copy_frame.manual != results_copy_frame.orig_manual] # Need a dataframe where the rows where both manual and orig_manual being None, have been droppoed

uncertain = pd.DataFrame()
for index, row in copy_frame_valid.iterrows():
    
    if uncertainty_check(row) is True:
        uncertain = uncertain.append(row)
        

# Remove Uncertain images from unknown
to_remove = [x for x in unknown.index if x in uncertain.index]
unknown = unknown.drop(to_remove, axis = 0)

# ... from Fairly_certain 
to_remove = [x for x in fairly_certain.index if x in uncertain.index]
fairly_certain = fairly_certain.drop(to_remove, axis = 0)

# ... from Certain
to_remove = [x for x in certain.index if x in uncertain.index]
certain = certain.drop(to_remove, axis = 0)

# See how many of the "unknown" images have at least one manual annotator agreeing with the ML label = ~50%
unknown_nones = unknown[(unknown.manual.isnull()) | (unknown.orig_manual.isnull())]
unknown_disagrees = unknown[(~unknown.manual.isnull()) & (~unknown.orig_manual.isnull())]

# Decided that we will combine Certain and Fairly certain into one final Certain category. 
# We also include a flag column. '1' indicate that the humans agreed with each other but Not the ML label
# and more importantly, flag == '0' means that the human(s) agreed WITH the ML label
certain = certain.append(fairly_certain)
certain['flag'] = '0'
certain['flag'].loc[fairly_certain.index] = '1'

# Final category overview
# Certain -> Both manual labelers agreed with each other and occasionally with the ML label    =  87.88% of images (9000) [33.83% agreed with the ML label, 66.16% did not agree]
# Unknown -> The manual labelers disagreed with each other (And maybe with the ML label)       =   7.72% of images (9000) [44.89% agreed with the Ml label, 55.10% did not agree]
# Uncertain -> The manual labelers used an "uncertainty" symbol                                =   4.38% of images (9000) 


# Final category overview when self-copy images have been removed
# Certain -> Both manual labelers agreed with each other and occasionally with the ML label    =  86.43% of images (7749) [33.83% agreed with the ML label, 66.16% did not agree] (Bit surprising that we got the same exact split between 'agreed with ML' or not, but seems to be the case)
# Unknown -> The manual labelers disagreed with each other (And maybe with the ML label)       =   8.96% of images (7749) [44.89% agreed with the Ml label, 55.10% did not agree]
# Uncertain -> The manual labelers used an "uncertainty" symbol                                =   4.69% of images (7749) 

# To see how many images where at least One reviewer agreed with the ML label in the Unknown frame = 3.97% of images (7749) and 44.83% of the Unknown images (687)
r1_agree = unknown[unknown.manual.isnull()]
r2_agree = unknown[unknown.orig_manual.isnull()]
at_least_one_agree = len(r1_agree) + len(r2_agree)

# This means that, if we were to accept these images as "correct" and add them to the Certain frame, that frame would then contain 6698 + 308 = 7006 images
# and would then give us a Certain "score" of 90.41%

# How many images in the Certain dataframe agreed with the Ml label? Answer: 2266 images
certain_agree_with_ml = certain[certain.manual.isnull()]

# Then, how many images in the Uncertain category agreed with the ML label? Answer: 64 images
r1_u_agree = uncertain[uncertain.manual.isnull()]
r2_u_agree = uncertain[uncertain.orig_manual.isnull()]
uncertain_agree_with_ml = len(r1_u_agree) + len(r2_u_agree)

# When we add up all images in the 3 dataframes where AT LEAST ONE reviewer agreed with the ML label
at_least_one_agree += len(certain_agree_with_ml)
at_least_one_agree += uncertain_agree_with_ml

# This gives us 2638 images of the entire dataset where AT LEAST ONE reviewer agreed with the ML label, which is 34.04%

