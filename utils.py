# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 14:58:20 2020

@author: Zoe
"""

import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def label_num2txt(pred):
    idx = np.argmax(pred)
    expressions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'neutral']
    label = expressions[idx]
    return label

def plotting_loss(epoch, train_loss, validate_loss, title):
    fig, axes = plt.subplots(1,1, figsize = (12, 8))
    axes.plot(epoch, train_loss, color = 'red', label = "Train")
    axes.plot(epoch, validate_loss, color = 'blue', label = "Validate")
    axes.set_title(title, fontsize = 25)
    axes.set_xlabel("Epochs", fontsize = 20)
    axes.set_ylabel("Loss", fontsize = 20)
    axes.grid()
    axes.legend(fontsize = 20)
    plt.savefig('output/loss_plot.jpg')

def plotting_acc(epoch, train_acc, validate_acc, title):
    fig, axes = plt.subplots(1,1, figsize = (12, 8))
    axes.plot(epoch, train_acc, color = 'red', label = "Train")
    axes.plot(epoch, validate_acc, color = 'blue', label = "Validate")

    axes.set_title(title, fontsize = 25)
    axes.set_xlabel("Epochs", fontsize = 20)
    axes.set_ylabel("Accuracy", fontsize = 20)
    axes.grid()
    axes.legend(fontsize = 20)
    plt.savefig('output/accuracy_plot.jpg')
    
    
def count_labels(Y_train, Y_validate):
    
    expressions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'neutral']
    y_pos = np.array(range(len(expressions)))
    
    if len(Y_train) > 1:
        num_in_label = np.sum(Y_train, axis=0)
        plt.bar(y_pos, num_in_label, align='center', alpha=0.5)
        plt.xticks(y_pos, expressions)
        plt.ylabel('Count')
        plt.title('The Occurrence of each Emotion in Training Data')
        plt.savefig('output/count_training.jpg')
        
    if len(Y_validate) > 1:
        num_in_label = np.sum(Y_validate, axis=0)
        plt.bar(y_pos, num_in_label, align='center', alpha=0.5)
        plt.xticks(y_pos, expressions)
        plt.ylabel('Count')
        plt.title('The Occurrence of each Emotion in Validate Data')
        plt.savefig('output/count_validate.jpg')
    

def predict_analysis(Y_test, pred):
    # wrong predict in each label
    wrong_idx = Y_test != pred
    wrong_pred = []
    
    for idx in range(len(wrong_idx)):
        if wrong_idx[idx]:
            wrong_pred.append(Y_test[idx])
            
    expressions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'neutral']
    y_pos = np.array(range(len(expressions)))
    
    unique, counts = np.unique(wrong_pred, return_counts=True)
    unique_all, counts_all = np.unique(Y_test, return_counts=True)
    
    err_precentage = [0]*len(expressions)
    
    for label in unique:
        err_precentage[label] = counts[label]/counts_all[label]
    
    plt.bar(y_pos, err_precentage, color = 'r', align='center', alpha=0.5)
    plt.xticks(y_pos, expressions)
    plt.ylabel('precentage')
    plt.title('The Error precentage of each Emotion in Test Data')
    plt.savefig('output/wrong_test.jpg')


    