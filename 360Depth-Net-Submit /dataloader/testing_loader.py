import os
import glob


def dataloader(filepath):
    upper_path = "upper/"
    lower_path = "lower/"

    all_files = [
        img for img in os.listdir(filepath + upper_path) if img.find('.jpg') > -1
    ]

    up_test = [filepath + upper_path + img for img in all_files]
    down_test = [filepath + lower_path + 'lower' + img[5:] for img in all_files]
    print("up size: ", len(up_test), len(up_test[0]))
    print(up_test)

    return up_test, down_test