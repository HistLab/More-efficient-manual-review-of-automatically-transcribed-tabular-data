# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:14:42 2020

@author: bpe043
"""

from layout import Ui_MainWindow

from PyQt5 import QtWidgets, QtCore, QtGui

import os
import sqlite3
import numpy as np
import math
from datetime import datetime
import time

class rhd_GUI(QtWidgets.QMainWindow, Ui_MainWindow):
    
    def __init__(self):
        super().__init__()
        
        #self = Ui_MainWindow()
        self.setupUi(self)
        
        # Boolean variable for when to cut off the next button
        self.stopper = False
        
        self.fasit_code = None
        
        
        # Connect the window to the database
        current_dir = os.path.dirname(os.path.realpath(__file__))
        path = os.path.join(current_dir, "database\\images.db")        
        self.conn = sqlite3.connect(path)
        self.c = self.conn.cursor()
        
        # Get images and labels from the database
        self.c.execute("SELECT Code FROM fields")
        labels = self.c.fetchall()
        labels = [code for t in labels for code in t]
        
        
        # Get unique labels, used for fasit code and to keep track of all the label values
        self.unique_labels = np.unique(labels)
        
        # Create a dictionary out of the labels.
        # The label itself is the key, and the number of images per label is the value. Ex. '531' : 100
        self.labels = {}
        for l in range(len(labels)):
            self.labels[labels[l]] = labels.count(
                    labels[l]
                    )
            
        # Create another dictionary that keeps track of how many images from each label we have "processed"
        # Used to keep track of if we've cycled through all the images for a specific label
        self.processed_labels = dict.fromkeys(self.labels, 0)
        
        # The first unique code we start with
        self.label_index = 0
        
        # We can use the number of codes in the database to calculate how many pages of images the user will see
        pages = [x / 60 for x in self.labels.values()]
        pages = [math.ceil(x) for x in pages]
        self.max_pages = sum(pages) + 1
        self.current_page = 0
        
        self.labelPagesDone.setText('Antall sider: {} av {}'.format(self.current_page, self.max_pages))
        
        
        # Setting maximum size for lables/images
        for lab in self.label_col:
            lab.setMaximumSize(QtCore.QSize(250, 120))
            
        # All line edits should be hidden at the start, and we also increase the font size
        font = QtGui.QFont()
        font.setPointSize(18)
        for le in self.lineEdit_col:
            le.hide()
            le.setFont(font)
            
        # Boolean class flag to indicate first time population vs. normal population
        self.init_over = False
        
        
        # When the Next button is hit
        self.updateButtonWindow2.clicked.connect(self.get_images)
        
        # TEST
        # When the Back button is hit
        self.backButton.clicked.connect(self.back)
        
    def back(self):
        
        # Process:
        # If you hit back as the first thing that's done, nothing should happen, same with hitting it at the very first page of images
        # Update the page counter to reflect going one page backwards
        # We should get the last 60 images (MAX) where the code is the previous code (If all previous codes are completed) or the current code (If not all current codes have been completed)
        # Once we know the fasit code status from the previous step, we need to update our fasit code
        # Then remove their verified status
        # Check for previously transcribed text
        # Display those images
        # Things run as normal when the next button is clicked again
        
        if self.fasit_code == None:
            return(None)
            
        if (self.current_page == 1):
            return(None)
            
        
        # Update the page counter
        self.current_page -= 1
        self.labelPagesDone.setText('Antall sider: {} av {}'.format(self.current_page, self.max_pages))
        
        # Get images, and check verification status on the code
        img_query = "SELECT Verified FROM fields WHERE Code == '{}'".format(self.fasit_code)
        self.c.execute(img_query)
        verified = self.c.fetchall()
        verified = [x[0] for x in verified]
        
        # If we need to go back with the CURRENT CODE
        if len(set(verified)) != 1:
            img_query = "SELECT Name, Image FROM fields WHERE Code == '{}' AND Verified == '1'".format(self.fasit_code)
            self.c.execute(img_query)
            data = self.c.fetchall()
            
            # Get the previous 60 images
            if len(data) > 60:
                data = data[-60:]
                
            # Update the tracker to remove these images
            self.processed_labels[self.fasit_code] -= len(data)
                
        # If we DO need to go back with the PREVIOUS CODE
        if len(set(verified)) == 1:
            # Remove the processed status of the "current" images that were left behind
            self.processed_labels[self.fasit_code] -= len(self.image_names)
            self.label_index -= 1
            self.fasit_code = self.unique_labels[self.label_index]
            
            img_query = "SELECT Name, Image FROM fields WHERE Code == '{}' AND Verified == '1'".format(self.fasit_code)
            self.c.execute(img_query)
            data = self.c.fetchall()
            
            # Get the previous 60 images
            if len(data) > 60:
                data = data[-60:]
                
            # Update Fasit code
            self.fasitLabelWindow2.setText(self.unique_labels[self.label_index])
    
            
            
        # Set the previous images as the ones to display
        self.image_names = [x[0] for x in data]
        self.images = [x[1] for x in data]
        
        # Remove Verification
        self.un_verify()
        
        # Clear current images
        self.clear_previous()
        
        # Check for previously transcribed text
        self.check_transcribed()

        # Display the previous images
        self.populate_images()
            
        
        
        
    def get_images(self):
        
        # First time initialization of the fasit code
        if self.fasitLabelWindow2.text() == "Fasit kode":
            self.fasitLabelWindow2.setText(self.unique_labels[self.label_index])
        
        # If we've gone through all the labels. Set a stopper value?
        if self.label_index >= len(self.unique_labels):
            self.stopper = True
            self.fasitLabelWindow2.setText("Ferdig!")
            return None
        
        if self.stopper == True:
            self.fasitLabelWindow2.setText("Ferdig!")
            return None
        
        # Update the page counter
        self.current_page += 1
        if self.current_page <= self.max_pages:
            self.labelPagesDone.setText('Antall sider: {} av {}'.format(self.current_page, self.max_pages))
        
        
        # Once the button has been pressed AFTER the first time, we need to check for input from the previous screen
        # update our database, and clear out the images and line edits from the previous screen
        if self.init_over == True:
        
            # We check for user input
            self.check_status()
            
            # Update the database
            self.update_image_status()
            
            # After which we clear the previous updates
            self.list_of_updates.clear()
            
            # And clear previous images
            self.clear_previous()
        
        
        # After we skip the above block once, we no longer need to
        self.init_over = True
        
        self.fasit_code = self.unique_labels[self.label_index]
        self.fasitLabelWindow2.setText(self.unique_labels[self.label_index])
        
        # If we have processed all the images for the given label, we should move on to the next
        if self.processed_labels[self.fasit_code] == self.labels[self.fasit_code]:
            self.label_index += 1
            
            if self.label_index >= len(self.unique_labels):
                self.stopper = True
                self.fasitLabelWindow2.setText("Ferdig!")
                return None
            
            self.fasit_code = self.unique_labels[self.label_index]

            # Update Fasit code
            self.fasitLabelWindow2.setText(self.unique_labels[self.label_index])
        
        img_query = "SELECT Name, Image FROM fields WHERE Verified == '0' AND Code == '{}' LIMIT 60".format(self.fasit_code)
        self.c.execute(img_query)
        data = self.c.fetchall()
        
        # Update the tracker with the amount of images we have now processed
        self.processed_labels[self.fasit_code] += len(data)
        
        # Check if these images have already been verified
        if len(data) == 0:
            
            # We also want to update the tracker with how many images we HAVE gone through
            # (This line might only be useful when starting fresh for a new session, but still... better safe than sorry. Because at this point I've forgotten what the tracker does)
            self.c.execute("SELECT Name, Image FROM fields WHERE Verified == '1' AND Code == '{}' LIMIT 60".format(self.fasit_code))
            data_2 = self.c.fetchall()
            self.processed_labels[self.fasit_code] += len(data_2)
            
            # Keep iterating over the labels until we've gone through them all
            if self.label_index < len(self.unique_labels):
                self.current_page = self.get_page_count()
                #self.label_index += 1
                self.label_index = self.get_label_index()
                self.init_over = False
                self.get_images()
                
                # The above call is a recursive call, to a method that ultimately returns nothing. 
                # This means that it *Probably* returns HERE, once it's "done". Then it executes the remaining code block, which can cause a crash because
                # it resets the image_names and images variables. Try to return something here instead
                return None
                
        # Else, we store the data from the images
        self.image_names = [x[0] for x in data]
        self.images = [x[1] for x in data]
        
        # Since we now have a Back function, we might have manually transcribed information for our images
        # To make sure that text stays, and is displayed for each image, even when flipping back and forth between pages, we need to check for that information
        self.check_transcribed()
        
        # We also update the image's Timestamp_star value, to note when the transcriber started working on these images
        self.start_timing()

        # And send them off to populate the screen
        self.populate_images()
        
        
    # Updating the start-time for the images that will be populating the screen
    def start_timing(self):
        
        # UNIX-time
        ts = time.time()
        
        # Translated to Norwegian datetime
        ts_actual = datetime.utcfromtimestamp(ts).strftime('%d-%m-%Y %H:%M:%S')
        
        # Then we use this datetime value to update the images we are working on
        for name in self.image_names:
            update_query = "UPDATE fields SET Timestamp_start = '{}' WHERE Name == '{}'".format(ts_actual, name)
            self.c.execute(update_query)

        
        # And we commit our changes to the database
        self.conn.commit()
        

        

    # Check for manually transcribed text already present
    def check_transcribed(self):
        
        index = 0
        for name in self.image_names:
            transcribed_query = "SELECT Manual FROM FIELDS WHERE Name == '{}'".format(name)
            transcribed = self.c.execute(transcribed_query).fetchone()
            transcribed = transcribed[0]
            
            if transcribed != None:
                le = self.lineEdit_col[index]
                le.setText(transcribed)
                
            index += 1
            
        
    
    # Checks to see if any image have been "corrected"
    def check_status(self):
        
        # The image that corresponds to each line edit
        image_index = 0
        
        # List of tupples containing the image names that need updating, and their new value
        self.list_of_updates = []
        
        # Check all the line edits for new text
        for le in self.lineEdit_col:
            if le.text() != "":
                img_name = self.image_names[image_index]
                new_label = le.text()
                
                self.list_of_updates.append((img_name, new_label))
                
            image_index += 1
            
        
    def un_verify(self):
        
        # When using the Back function, we also remove the "Verified" status of the images
        for name in self.image_names:
            verify_query = "UPDATE fields SET Verified = '0' WHERE Name == '{}'".format(name)
            self.c.execute(verify_query)

        
        # And we commit our changes to the database
        self.conn.commit()
        
        
    def get_label_index(self):
        
        label_query = "select distinct Code from fields where Verified = '1'"
        completed = self.c.execute(label_query).fetchall()
        
        label_index = len(completed)
        
        return int(label_index)
        
        
    def get_page_count(self):
        
        page_query = "select Code from fields where Verified = '1'"
        completed = self.c.execute(page_query).fetchall()
        completed = [x[0] for x in completed]
        uniques = np.unique(completed)
        
        pages = 0
        for code in uniques:
            count = completed.count(code)
            
            if count > 60:
                i = math.ceil(count / 60)
                pages += i
            else:
                pages += 1
        
        return pages
        
        
    # Sets an image status as Verified in the database
    def update_image_status(self):
        
        # Check to see if there are any images that need to be updated
        if len(self.list_of_updates) != 0:
            for updates in self.list_of_updates:
                img_name = updates[0]
                new_label = updates[1]
                
                update_query = "UPDATE fields SET Manual = '{}' WHERE Name == '{}'".format(new_label, img_name)
                self.c.execute(update_query)
        
        # We also check off all the images, indicating that they have been looked at        
        for name in self.image_names:
            verify_query = "UPDATE fields SET Verified = '1' WHERE Name == '{}'".format(name)
            self.c.execute(verify_query)
            
        # Additionally, we update the images' Timestamp_end value with the current time. 
        # Together with the start_timing function, this will allow us to measure the time used per page of images
        
        # UNIX-time
        ts = time.time()
        
        # Translated to Norwegian datetime
        ts_actual = datetime.utcfromtimestamp(ts).strftime('%d-%m-%Y %H:%M:%S')
        for name in self.image_names:
            update_query = "UPDATE fields SET Timestamp_end = '{}' WHERE Name == '{}'".format(ts_actual, name)
            self.c.execute(update_query)
        
        # And we commit our changes to the database
        self.conn.commit()
        
        
        
    def populate_images(self):
        
        nr_images = len(self.images)
        
        for i in range(nr_images):
            
            img = self.images[i]
            pixmap = self.get_pixImg(img)   
            
            # Assign the pixmap to a label, and show it
            self.label_col[i].setPixmap(pixmap)
            self.label_col[i].show()
            
            # Show the checkbox for each image
            self.lineEdit_col[i].show()
            
            i += 1
            
        
         
            
    def get_pixImg(self, imageBytes):
        ba = QtCore.QByteArray(imageBytes)
        qimg = QtGui.QImage.fromData(ba)
        pixmap = QtGui.QPixmap.fromImage(qimg) 
        
        return pixmap
    
    def clear_previous(self):
        # Clear labels
        for lab in self.label_col:
            lab.clear()

        # Clear and hide line edits
        for le in self.lineEdit_col:
            le.clear()
            le.hide()
        
        
    def keyPressEvent(self, event):
        
        if event.key() == QtCore.Qt.Key_Escape:
            focused_widget = QtWidgets.QApplication.focusWidget()
            
            if focused_widget != None:
                QtWidgets.QApplication.focusWidget().clearFocus()
                
                
        if event.key() == QtCore.Qt.Key_Right:
            self.get_images()
        
    def mousePressEvent(self, event):
        focused_widget = QtWidgets.QApplication.focusWidget()
        
        if focused_widget != None:
            focused_widget.clearFocus()
        
        
        
if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    
    mainWindow = rhd_GUI()
    mainWindow.show()
    
    app.exec_()
        
