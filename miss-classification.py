# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 10:44:38 2023

@author: bpe043
"""

import pandas as pd
import sqlite3
import numpy as np

def separate_invalids(x):
    
    # If there are 3 characters, but they are NOT 'bbb' or 'ttt', we return them
    if len(x) == 3:
        if not x.isnumeric():
            if x != 'bbb' and x != 'ttt':
                return True
        
    # If it has any other length, we return it
    else:
        return True
            

# Load in the data
data = pd.read_sql("select * from fields", con = sqlite3.connect("results.db"))
data.columns = [x.lower() for x in data.columns]
data = data[['name', 'code', 'manual']]

# Remove the copy-images from the data
copy_data = data[data['name'].str.endswith('copy.jpg')]
data = data.drop(copy_data.index)
del copy_data

# Remove leading or trailing whitespace in the 'code' and 'manual' fields
data['code'] = data['code'].apply(lambda x: x.strip())
data['manual'] = data['manual'].apply(lambda x: x.strip() if x is not None else x)
data_orig = data.copy()

# Human labels are treted as "the true code", and will determine the number of images for each code
# The ML label can either be the same as the human label, or different.
# Let's split this into a frame with Only humans and ML agreeing, and Only humans and ML disagreeing

# Find all images where the reviewers had absolutely no idea what was on the image, i.e they wrote down only "??" as the label
no_clue = data[data.manual == '??']

# Note that before we do that though, we should remove the images where the label is "functionally un-knowable"
# These are the images where the human used one of the "Uncertainty" symbols we defined in the user guide. Without looking at these images, we cannot know for sure what is in them (5% of all images [90,000])
uncertainty_symbols = '|'.join(['\?', '@', '%'])
uncertains = data[data.manual.str.contains(uncertainty_symbols, na = False)]
data = data.drop(uncertains.index)

# Where the labels agree
# Here, this is usually denoted by the human label being None (28,545 instances, or 31.72% of all images [90,000])
# but there are Some instances where the human entered the same label as the ML code (219 instances, or 0.24% of all images [90,000]) 
# [So, roughly 32% of all images had a humand and the ML agreeing on the label]
agree = data.loc[data.manual.isnull()]
agree_same = data.loc[data.manual == data.code]
agree = pd.concat([agree, agree_same])
data = data.drop(agree.index)

# Sometimes, the humans missed out on some codes, and agreed when it was obvious that this code cannot exist, such as '55b' (Or they do think that is actually what's on the image)
# Either way, we can't know for sure, so we remove those labels from the agree frame (This was the case for 16 instances, or 0.05% of the agree images [28,764 images] and 0.01% of all images [90,000])
agree_invalids = agree.code.apply(lambda x: separate_invalids(x))
agree_invalids = agree_invalids[agree_invalids == True]
agree = agree.drop(agree_invalids.index)

# Another (bonus) category is where the human labeler made a mistake, either writing two labels in one box, hitting the wrong key when going to enter a character (such as + instead of ?), and so forth
# We also separate these out (181 codes, or 0.20% of all images [90,000])
invalids = data.manual.apply(lambda x: separate_invalids(x))
invalids = invalids[invalids == True]
data = data.drop(invalids.index)

# Where the labels disagree
# This is where the humand and ML label differs and the human label is NOT None (56,557 instances, or 62.84% of all images [90,000])
# (This is really all that should remain, but why not do the check just to be sure)
disagree = data.loc[(data.manual != data.code) & (~data.manual.isnull())]

# Now that we have all our frames, we can start to calculate the number of missclassifications and correct classifications

# To get the number of correct classifications by the ML per code
# We can't keep working with Nones, so we instead overwrite the None value with the 'code' value. We know that this is what the None value really means after all
agree['manual'].loc[agree.manual.isnull()] = agree['code']

# Then we need to find a number for each of these labels
agree = agree.groupby('manual')['code'].size().to_frame('ml_label_correct_count')
agree['true_label'] = agree.index
agree = agree.sort_values(by = 'ml_label_correct_count', ascending = False).reset_index(drop = True)

# We see, for the disagree frame images, how many times the humans assigned each label
disagree_humans = disagree['manual'].value_counts().to_frame('ml_label_incorrect_count')
disagree_humans['true_label'] = disagree_humans.index
disagree_humans = disagree_humans.reset_index(drop = True)

""" I don't think we actually need the frame below. In disagree_humans, we alredy have an overview of images per true (ref. human) label that the ML was wrong about, by definition. """
# Then we see, for the same set of images, how many times the ML assigned an incorrect label to the image
disagree_ml = disagree['code'].value_counts().to_frame('count')
disagree_ml['ml_label_incorrect'] = disagree_ml.index
disagree_ml = disagree_ml.reset_index(drop = True)


# We merge the two frames (correct and incorrect by true label) to gather together the count values.
# Where there were no correct classifications, we set 0
merged = disagree_humans.merge(agree['ml_label_correct_count'], how = 'left', left_on = 'true_label', right_on = agree['true_label'])
merged.fillna(0, inplace = True)
merged['n'] = merged['ml_label_incorrect_count'] + merged['ml_label_correct_count']
merged = merged.sort_values(by = 'n', ascending = False).reset_index(drop = True)

# Calculate the percentages of wrongly/correctly classified labels
merged['perc_wrong_classification'] = ((merged['ml_label_incorrect_count'] / merged['n']) * 100).round(2)
merged['perc_right_classification'] = ((merged['ml_label_correct_count'] / merged['n']) * 100).round(2)


""" Time for plotting!! """

import seaborn as sns
import matplotlib.pyplot as plt
import smooth

merged_p = merged[['true_label', 'perc_wrong_classification', 'perc_right_classification', 'n']]


# Getting the list of official codes from Statistics Norway - Want to try and limit merged_p to only the codes found in here, and see what happens to the figure
code_list = pd.read_csv('1950_Occupational_Codes_list.csv', sep = ';', encoding = 'latin-1')
code_list.columns = [x.lower() for x in code_list.columns]
code_list = code_list['code'].tolist()

#merged_p = merged_p[merged_p['true_label'].isin(code_list)]

    
# Scatterplot

# We also created a smoothed regression curve, in order to do this, we need numerical values for both the x and y axis,
# and since the x-axis is an index of all classes, we just use the raw index as the x-axis
merged_p['points'] = merged_p.index
pred_vals = smooth.plot_rcs(merged_p, 'points', 'perc_wrong_classification', num_knots=6, ret_data = True, plot = False)

fig, ax = plt.subplots()
sns.lineplot(x = pred_vals['x'], y = pred_vals['mean'], 
        zorder=3, color='darkred',alpha=0.9, ax = ax)
ax.fill_between(pred_vals['x'],pred_vals['mean_ci_lower'],
                pred_vals['mean_ci_upper'],alpha=0.2,
                zorder=2, color='red')
sns.scatterplot(x = merged_p['points'], y = merged_p['perc_wrong_classification'], 
           color = 'lightblue', edgecolor = 'k',
           s=15, alpha=0.7, 
           zorder=4, ax = ax)

sns.despine(offset = {'left' : 10, 'bottom' : 0})
ax.set(ylim = (0, 115))
ax.set(xlim = (-5, len(merged_p) + 5))       # Scatter plot
ax.set(xticklabels = [])
label_placements = [-5, 137, 274, 411, 554] # Placing a tick mark on the x-axis at min, max (slightly adjusted with +\-5 for aesthetics), at the median, and the 25% and 75% marks
label_text = [str(int(merged_p['n'].loc[0])), str(int(merged_p['n'].loc[label_placements[1]])), str(int(merged_p['n'].loc[label_placements[2]])), str(int(merged_p['n'].loc[label_placements[3]])), str(int(merged_p['n'].loc[len(merged_p)-1]))]
ax.set_xticks(ticks = label_placements, labels = label_text)

ax.set_title('3% dataset\nModel classification error by class size', pad = 15)
ax.set_ylabel('Missclassification percentage')
ax.set_xlabel("Class size")
ax.figure.savefig('scatterplot_missclassification_rate_regressionLine_allCodes_testing.png', dpi = 300, bbox_inches = 'tight')

plt.show()




# Look at specific missclassifications for the codes with the highest missclassification rate

from wordcloud import WordCloud

most_wrong = merged_p[merged_p['perc_wrong_classification'] > 80].sort_values('perc_wrong_classification', ascending = False).reset_index(drop = True)
most_wrong.insert(3, 'ml_labels', None)

for index, row in most_wrong.iterrows():
    ml_labels = disagree[disagree.manual == row['true_label']]
    ml_labels = ml_labels.code.value_counts().to_dict()
    most_wrong.at[index, 'ml_labels'] = ml_labels
    
    
# Even for the most frequent codes, the same pattern of similarity is a stronger indication for which codes were mistaken for the proper one
most_frequent_codes = ['531', '899', '555', '111']
most_frequent = merged_p[merged_p['true_label'].isin(most_frequent_codes)].reset_index(drop = True)
most_frequent.insert(3, 'ml_labels', None)

for index, row in most_frequent.iterrows():
    ml_labels = disagree[disagree.manual == row['true_label']]
    ml_labels = ml_labels.code.value_counts().to_dict()
    most_frequent.at[index, 'ml_labels'] = ml_labels
    

test = most_frequent.loc[3]['ml_labels']
wordcloud = WordCloud(width = 1000, height = 1000, max_words = 15, background_color = 'white', colormap = 'twilight_shifted').generate_from_frequencies(test)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud)
plt.axis('off')
plt.title("Misclassifications for the code text", pad = 20)
#plt.savefig(f'misclassification_example_text.png', dpi = 300)

""" See if the top X wrong classifications from the model corresponds to the biggest classes, as defined by the distribution of classes in the training set """
import os

# Get the distribution from the training set used to train the model
temp = []
for root_dir, cur_dir, files in os.walk(r'C:\Users\bpe043\OneDrive - UiT Office 365\Papers\OCCODE\RandomSample_trainingset'):
    temp.extend(files)
    
training_dist = pd.DataFrame(columns = ['label'], data = [x.split('-')[0] for x in temp])
training_dist['count'] = training_dist['label'].map(training_dist['label'].value_counts(normalize = False))
training_dist = training_dist.drop_duplicates('label').sort_values(by = 'count', ascending = False).reset_index(drop = True)


del temp, root_dir, cur_dir, files


wrong_guesses = disagree_ml.iloc[0:10]
wrong_guesses.rename(columns = {'ml_label_incorrect' : 'label'}, inplace = True)

ax = sns.barplot(data = wrong_guesses, x = 'label', y = 'count', color = 'lightblue')
sns.despine(offset = {'left' : 10, 'bottom' : 0})
#ax.set(ylim = (0, 1.1))
ax.set_title('Top 10 erroneous classifications by the ML model')
ax.set_ylabel('Count')
ax.set_xlabel('Label')
#ax.figure.savefig('10_most_frequent_classes_ML_labels_unScaled.png', bbox_inches = 'tight', dpi = 300)
ax.clear()


training_dist = training_dist.iloc[0:10]

ax_train = sns.barplot(data = training_dist, x = 'label', y = 'count', color = 'lightblue')
sns.despine(offset = {'left' : 10, 'bottom' : 0})
#ax_train.set(ylim = (0, 1.1))
ax_train.set_title('10 largest classes in the training set')
ax_train.set_ylabel('Count')
ax_train.set_xlabel('Label')
#ax_train.figure.savefig('10_most_frequent_classes_training_unScaled.png', bbox_inches = 'tight', dpi = 300)


""" We can see from various codes above that the biggest classes are the ones that were Incorrectly guessed the most. Let's try to find where they were guessed the most """
most_frequent_codes = ['531', '899', '555', '111']

for code in most_frequent_codes:
    
    x = disagree[disagree.code == code].manual.value_counts()
    x = x.iloc[0:10]
    x = x.to_frame(name = 'count')
    x['label'] = x.index
    
    ax = sns.barplot(data = x, x = 'label', y = 'count', color = 'lightblue')
    ax.set(ylim = (0, 350))
    ax.set_title('Human correction of model classification for code {}'.format(code))
    ax.set_ylabel('Count')
    ax.set_xlabel('Human corrected label')
    #ax.figure.savefig('corrections_for_most_frequent_code_{}.png'.format(code), bbox_inches = 'tight', dpi = 300)
    ax.clear()
    
    
""" In the same vein, the code above shows us the corrections for when the model misclassified an image as one of the big, VALIUD, classes. What were actually on the images where it said something Invalid? """
most_frequent_invalids = ['ttb', 'bb1', '55b', '5bb']

for invalid in most_frequent_invalids:

    x = disagree[disagree.code == invalid].manual.value_counts()
    x = x.iloc[0:10]
    x = x.to_frame(name = 'count')
    x['label'] = x.index
    
    ax = sns.barplot(data = x, x = 'label', y = 'count', color = 'lightblue')
    
    if invalid == 'bb1':
        ax.set(ylim = (0, 1000))
    else:
        ax.set(ylim = (0, 700))
    ax.set_title('Human correction of model classification for code {}'.format(invalid))
    ax.set_ylabel('Count')
    ax.set_xlabel('Human corrected label')
    ax.figure.savefig('corrections_for_most_frequent_invalid_{}.png'.format(invalid), bbox_inches = 'tight', dpi = 300)
    ax.clear()
    
    
""" See the Ml classifications for images with 'bbb' or 'ttt' """
blank_and_texts = disagree[(disagree.manual == 'bbb') | (disagree.manual == 'ttt')].code.value_counts().to_frame('count')
blank_and_texts['label'] = blank_and_texts.index
blank_and_texts.reset_index(drop = True, inplace = True)
blank_and_texts['cumsum'] = 100 * (blank_and_texts['count'].cumsum() / blank_and_texts['count'].sum())


""" We find that if we limit ourselves to only the codes found in code_list, we see a trend of misclassification being most prevalent amongst the biggest classes. Is this because the biggest classes have some of the most common digits? """
code_list_single_digits = []
for code in code_list:
    for character in code:
        code_list_single_digits.append(character)
        
# One code in the official list of codes contains characters, we remove this code
del code_list_single_digits[285:288]

code_list_single_digits_values, code_list_single_digits_counts = np.unique(code_list_single_digits, return_counts = True)
code_list_single_digits_frame = pd.DataFrame()
code_list_single_digits_frame['digit'] = code_list_single_digits_values
code_list_single_digits_frame['count'] = code_list_single_digits_counts
code_list_single_digits_frame = code_list_single_digits_frame.sort_values(by = 'count').reset_index(drop = True)
