# -*- coding: utf-8 -*-
"""
Created on Tue Jan 11 10:27:24 2022

@author: bpe043
"""

""" 
    For each user, there is a folder called <User>/review/images where the images they have been assigned have been placed
    These images needs to be converted into a database of images for the review tool to work.
    This database needs to have a table called 'fields' and the following columns: Name, Image (blob), Verified (default '0'), Code, Manual, Timestamp_start, Timestamp_end
    Code is the field for the label given by the machine learning model
    Manual is where the reviewers will input the correct label (if needed)
"""

import os
import cv2
import sqlite3

    
def create_db(cur):
    
    cur.execute(""" create table if not exists fields
                (
                Name TEXT,
                Image BLOB,
                Verified TEXT DEFAULT 0,
                Code TEXT,
                Manual TEXT,
                Timestamp_start TEXT,
                Timestamp_end TEXT
                )
                """)


main_path = 'Users/'

users = os.listdir(main_path)

# Iterate over all users
for user in users:
    
    # Each user needs a database
    usr_db = sqlite3.connect(main_path + user + '/database/images.db')
    cur = usr_db.cursor()
    create_db(cur)

    # Get images from the review folder
    for img in os.listdir(main_path + user + '/images'):
        
        name = img.split('-')[1]
        code = img.split('-')[0]
        img_bytes = cv2.imread(main_path + user + '/images/' + img)
        img_bytes = cv2.imencode('.jpg', img_bytes)[1].tobytes()
        
        cur.execute('insert into fields(Name, Image, Code) values (?, ?, ?)', (name, img_bytes, code))
        usr_db.commit()
    
    usr_db.close()
    
