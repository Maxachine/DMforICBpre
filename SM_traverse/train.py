from train_fn import train_fn
import os

if __name__ == '__main__':
    train_folder = "train_files" 

    for file_name in os.listdir(train_folder):
        if file_name.endswith(".txt"): 
            train_fn(file_name) 