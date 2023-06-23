# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 09:48:25 2022

@author: bpe043
"""

import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import numpy as np

from matplotlib.ticker import StrMethodFormatter

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter

OUT_PATH = "paper_images\\results\\"


# Function to plot a cumulative sum curve for each user 
def plot_time_series(frame, user, t = 'start'):
    
    # Get the total and average number of hours spent
    #tot, avg, _ = return_time_usage(frame, user)
    
    # Select the desired time series
    if t == 'start':
        frame = frame[['user', 'timestamp_start']].reset_index(drop = True)
    else:
        frame = frame[['user', 'timestamp_end']].reset_index(drop = True)
    
    # Since all of these transcriptions were done within the span of 2 months, we are only interested in the first 5 characters
    frame['timestamp_start'] = frame['timestamp_start'].str[:5]
    
    # We want to see how many images the user went throuh per time points
    frame['sum'] = frame.groupby(['timestamp_start']).transform('count')
    frame = frame.drop_duplicates('timestamp_start')
    frame = frame[['timestamp_start', 'sum']].reset_index(drop = True)
    
    # We add 1 to the index because you can't work for 0 work days
    frame.index = frame.index + 1
    #frame.reset_index = frame['timestamp_start']
    
    # And add together the number of images per point, to get a nice curve
    frame = frame['sum'].cumsum()
    
    # Then we plot
    fig = frame.plot(xticks = frame.index, figsize = (8, 6))
    fig.set_xlabel("Work days")
    fig.set_ylabel("Number of images", labelpad = 10)
    fig.set_title("Number of work days used to review all images for user " + user)
    plt.show()
    
    # Save the plot
    name = OUT_PATH + user + '_plot.png'
    fig.get_figure().savefig(name)
    
    
""" 
    Function that takes in the start and end time (Only Hours:Minutes:Seconds, not days) and converts the difference between them to seconds
    s1 = end_time
    s2 = start_time
"""
def convert_to_seconds(s1, s2):

    # If s1 or s2 is None, return -1 and we will remove it later
    if s1 is None or s2 is None:
        return -1
    
    fmt = '%H:%M:%S'
    diff = datetime.strptime(s1, fmt) - datetime.strptime(s2, fmt)
    diff = int(diff.total_seconds())
    
    return diff


def plot_custom(x_line, name, x_label, y_label, title, y_max, y_min = -2, mean = False):
    
    # Despine
    x_line.spines['right'].set_visible(False)
    x_line.spines['top'].set_visible(False)
    x_line.spines['left'].set_visible(False)

    
    if mean:
        # Add in the mean-line
        x_line.axhline(mean, color = 'r', zorder = 2, label = 'Avg. time use')

    # Switch off ticks
    x_line.tick_params(axis="both", which="both", bottom= False, top= False, labelbottom="on", left= False, right= False, labelleft="on")
    
    # Draw horizontal axis lines
    vals = x_line.get_yticks()
    for tick in vals:
        x_line.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)
        
    # Set the y limit
    x_line.set_ylim(y_min, y_max)
        
    # Set title
    x_line.set_title(title)

    # Set x-axis label
    x_line.set_xlabel(x_label, labelpad=20, weight='bold', size=12)

    # Set y-axis label
    x_line.set_ylabel(y_label, labelpad=20, weight='bold', size=12)

    # Format y-axis label
    x_line.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    
#    handles, labels = x_line.get_legend_handles_labels()
#    x_line.legend(handles[1:], labels[1:])
    
    # Save
    x_line.figure.savefig('plots/' + name, dpi = 300)
    
    plt.show()
    
    # Clear the figure
    #x_line.clear()
    
   
def plot_hist(frame, title, name):
    
    ax_hist = frame.hist(column='seconds_per_image_per_code', bins= 25, grid=False, figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)
    x_hist = ax_hist[0][0]
    
    # Despine
    x_hist.spines['right'].set_visible(False)
    x_hist.spines['top'].set_visible(False)
    x_hist.spines['left'].set_visible(False)

    # Switch off ticks
    x_hist.tick_params(axis="both", which="both", bottom= False, top= False, labelbottom="on", left= False, right= False, labelleft="on")
    
    # Draw horizontal axis lines
    vals = x_hist.get_yticks()
    for tick in vals:
        x_hist.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)
        
    # Set title
    x_hist.set_title(title)

    # Set x-axis label
    x_hist.set_xlabel("Nr. seconds", labelpad=20, weight='bold', size=12)

    # Set y-axis label
    x_hist.set_ylabel("Nr. images", labelpad=20, weight='bold', size=12)

    # Format y-axis label
    x_hist.yaxis.set_major_formatter(StrMethodFormatter('{x:,g}'))
    
    # Save
    x_hist.figure.savefig('plots/' + name)
    
    plt.show()
    
    # Clear the figure
    #x_hist.clear()
    
    
def plot_line(frame, name, user):
    
    # We also want to add a mean-line to the plot
    mean = frame['seconds_per_image_per_code'].mean()
    
    ax_line = frame['seconds_per_image_per_code'].plot( kind = 'line', grid=False, figsize=(12,8), zorder=2)
    x_line = ax_line
    
    

def plot_savgol_multiple(list_of_frames):

    con_frame = pd.DataFrame()
    
    # The x axis corresponds to the number of unique codes that a user had. This can vary, so we need to find the longest x
    x = None
    
    # We also find the biggest y value, so we can set the size limit in the plot
    y_max = None
    
    for f in list_of_frames:
        
        # Find x
        if x is None:
            x = len(f)
        else:
            if len(f) > x:
                x = len(f)
                
        # Then we want to take the 'seconds_per_image_per_code' column from f, add the user as part of the column name, and then add it to the 'con_frame'
        f = f.reset_index(drop = True)
        user = f['user'].loc[0]
        y = pd.Series(savgol_filter(f['seconds_per_image_per_code'], 51, 3))
        y.name = 'Reviewer ' + user
        con_frame[y.name] = y
        
        # Find y_max
        if y_max is None:
            y_max = y.max()
        else:
            if y.max() > y_max:
                y_max = y.max()
        
    # Longest x becomes the plot "index"
    x_final = np.arange(x)
    con_frame['ind'] = x_final
    
    # Create the axes object
    ax = con_frame.plot(x = 'ind', y = list(con_frame.columns[0:-1].values), kind = 'line', figsize = (17, 8))
    plot_custom(ax, 'Smooth_plot_all_users_2.png', 'Nr. unique codes', 'Nr. seconds', 'Seconds per image per code for each reviewer', y_max+1, 0)
    

def plot_savgol(plot_frame, name, user):
    
    plot_frame = plot_frame.reset_index(drop = True)
    
    x = plot_frame.index.values
    y = plot_frame['seconds_per_image_per_code'].values
    y_savgol = savgol_filter(plot_frame['seconds_per_image_per_code'], 51, 3)
    
    temp_frame = pd.DataFrame()
    temp_frame['x_vals'] = x
    temp_frame['seconds'] = y
    temp_frame['seconds_savgol'] = y_savgol
    
    ax = temp_frame.plot(x = 'x_vals', y = ['seconds', 'seconds_savgol'], kind = 'line', figsize = (12, 8), legend = False, color = ['tab:blue', 'r'])
    
    y_max = plot_frame['seconds_per_image_per_code'].max() + 2
    plot_custom(ax, "user_{}_seconds_used_smooth".format(user), "Nr. images", "Nr. seconds", "Seconds per image", y_max)
    
    
   
    
""" 
    Gets the frame of the individual user and calculates the amount of hours spent reviewing all the images
    as well as the average minutes spent per page
"""    
def return_time_usage(frame, user):
    
    #frame = frame.drop('user', axis = 1).reset_index(drop = True)
    
    frame[['start_day', 'start_time']] = frame['timestamp_start'].str.split(' ', 1, expand = True)
    frame[['end_day', 'end_time']] = frame['timestamp_end'].str.split(' ', 1, expand = True)
    
    frame['diff'] = frame.apply(lambda x: convert_to_seconds(x.end_time, x.start_time), axis = 1)


    # Removing the pages where there were the user took obvious longer breaks (And where the user might have accidentally skipped a page)
    frame = frame[(frame['diff'] < 3600) & (frame['diff'] > -0.1)]
    
    # We also want to see how long each user spent per ML code.
    unique_codes = frame.code.drop_duplicates().to_list()
    #code_frame = frame.drop_duplicates(subset = ['start_day', 'start_time']).reset_index(drop = True)
    
    #For each code, sum up the diff and store code-diff pairs in a new dataframe
    code_frame = pd.DataFrame(columns = ['code', 'diff', 'nr_imgs', 'freq'])
    for code in unique_codes:
        cf = frame[frame.code == code]
        nr_imgs = len(cf)
        freq = str(round((len(cf) / len(frame)) * 100, 2)) + '%'
        cf = cf.drop_duplicates(subset = ['start_day', 'start_time'])
        diff = cf['diff'].sum()
        
        x = {'code' : code, 'diff' : diff, 'nr_imgs' : nr_imgs, 'freq' : freq}
        
        code_frame = code_frame.append(x, ignore_index = True)
        
        
    code_frame['seconds_used'] = code_frame['diff'].astype(int)
    code_frame = code_frame.drop('diff', axis = 1)
    
    # Find the time usage per image per code
    code_frame['nr_imgs'] = code_frame['nr_imgs'].astype(int)
    code_frame['seconds_per_image_per_code'] = code_frame['seconds_used'] / code_frame['nr_imgs']
    
    """ Plot the time usage in seconds per image per code """
    # Including a separate plot for the codes that took an unusually long amount of time
    # For the images in the second plot, we also create a csv file with the information about the codes
    
    std = code_frame['seconds_per_image_per_code'].std()
    
    # Find the codes that took an unusual amount of time (Given that we already removed the pages where the user obviously took a break)
    deviation_frame = code_frame[code_frame['seconds_per_image_per_code'] > 3*std]
    
    # Remove those from the original frame and store it as a separate frame
    # So we can still use the original frame further down
    time_frame_hist = code_frame.copy()
    time_frame_hist = time_frame_hist.drop(deviation_frame.index)
    
    # Do the plots
    
    # For the images where the time usage was within 3 standard deviations
    title = 'Seconds used per image per code for user {}\nFor codes where the time usage was shorter than 3 STDs'.format(user)
    name = 'User_{}_hist.png'.format(user)
    plot_hist(time_frame_hist, title, name)
    
    # For the images where the time usage was longer than that
    if len(deviation_frame) > 0:
        title = 'Seconds used per image per code for user {} \nFor codes where the time usage was longer than 3 STDs'.format(user)
        name = 'User_{}_deviation_frame.png'.format(user)
        plot_hist(deviation_frame, title, name)
        deviation_frame.to_csv('plots/user_{}_deviation_frame.csv'.format(user), index = False, sep = ';')
    

    """ Plot the usage of time per image over time """
    # i.e. Did they spend more time on the images in the beginning, the end, or fairly balanced?
    time_frame_line = code_frame.copy()
    time_frame_line = time_frame_line.drop(deviation_frame.index)
    
    """ Smoothing test """
    #plot_savgol(code_frame, 'test', user)
    savgol_frame = code_frame.drop(deviation_frame.index)
    savgol_frame['user'] = user
    
    
    """ Smoothing test over """
    
    name = 'User_{}_time_used_line.png'.format(user)
    plot_line(time_frame_line, name, user)

    
    # Add in the value of how big the code was in the frame. So, how many images were there of that code for this user
    code_frame_2 = code_frame.copy()
    code_frame_2['freq'] = frame['code'].value_counts(normalize = True) * 100
    
    # We make the frame unique per timestamp, so we don't count the time for Each image per page, but rather per Page
    frame = frame.drop_duplicates('timestamp_start')
    
    # Then we can sum all the seconds up, and convert to hours.
    
    # If we want the hours represented as a float number with 2 decimals
    frame_tot = round(frame['diff'].sum() / 3600, 2)
    
    # If we want hours, minutes, and seconds instead
    #tot_seconds = frame['diff'].sum()
    #minutes, seconds = divmod(tot_seconds, 60)
    #hours, minutes = divmod(minutes, 60)
    
    avg = round((frame_tot * 60) / len(frame), 2)
    
    frame_tot = str(frame_tot) + ' total hours spent'
    avg = str(avg) + ' minutes per page'
    
    return frame_tot, avg, code_frame, savgol_frame



frame = pd.read_sql("select * from fields", con = sqlite3.connect("results.db"))
frame.columns = [frame.lower() for frame in frame.columns]
frame = frame[['user', 'code', 'timestamp_start', 'timestamp_end']]

a = frame[frame.user == 'a']
b = frame[frame.user == 'b']
c = frame[frame.user == 'c']
d = frame[frame.user == 'd']
e = frame[frame.user == 'e']
f = frame[frame.user == 'f']
g = frame[frame.user == 'g']

user_frames = [a, b, c, d, e, f, g]
users = ['a', 'b', 'c', 'd', 'e', 'f', 'g']

time_storage = []

# List to store the frames that will be used in the savgol (smooth) plotting
savgol_storage = []

i = 0
while i < len(user_frames):
    frame = user_frames[i]
    user = users[i]
    
    #plot_time_series(frame, user)
    frame_tot, avg_mins_per_page, seconds_per_code, savgol_frame = return_time_usage(frame, user)
    time_storage.append([frame_tot, avg_mins_per_page, seconds_per_code])
    savgol_storage.append(savgol_frame)
    #time_storage.append(return_time_usage(frame, user))
    i += 1
    print("Finished with user " + user)


plot_savgol_multiple(savgol_storage)