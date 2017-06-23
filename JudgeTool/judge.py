from pathlib import Path
import random
import sys
import cv2
import matplotlib as plt
import numpy as np
from PIL import Image
#import matplotlib.pyplot as plt
import os




def do_label(unjudgedlist, ofstr):
    judged_file = open(ofstr, 'a')
    for line in unjudgedlist:

        line = line.strip()
        cols =line.split(",")
        try:
            sid = cols[0]
            pic1 = cols[1]
            pic2 = cols[2]
            label = cols[3]
            k = show_img(pic1, pic2)
            if k  is None:
                print ("error input:", line)
                continue
            if k == 27:
                break
            if k == ord('y'):
                judged_file.writelines(line + ",1\n")
            if k == ord('n'):
                judged_file.writelines(line + ",0\n")
        except:
            print("error")
            continue
    judged_file.close()



# read file:
# sid, pic1, pic2, ..., ...,
# return dic: key = "sid, pic1, pic2", value = line
def load_pair_dic(fstr):
    dic = {}
    if not os.path.exists(fstr): return dic
    with open(fstr, "r") as f:
            for line in f:
                line = line.strip()
                cols = line.split(",")
                if len(cols)< 3: continue
                key = ",".join(cols[:3])
                dic[key] = line
    f.close()
    return dic

# get unjudged list
def get_unjudged_list(raw_list_fstr, judged_list_fstr):
    l = []
    raw_dic = load_pair_dic(raw_list_fstr)
    judged_dic = load_pair_dic(judged_list_fstr)
    for k in raw_dic.keys():
        if k in judged_dic:
            continue
        l.append(raw_dic[k])
    return l

def show_img(imgfstr1, imgfstr2, path = None):
    default_path = "E:\work@bing\linkedin\img_data\images\\"
    if path == None:
        path = default_path

    k = None
    imgfstr1 = path + imgfstr1
    imgfstr2 = path + imgfstr2
    if not os.path.exists(imgfstr1): return k
    if not os.path.exists(imgfstr2): return k


    img1 = cv2.imread(imgfstr1)
    img2 = cv2.imread(imgfstr2)

    cv2.imshow('image1', img1)
    cv2.imshow('image2',img2)

    cv2.moveWindow('image2', 200, 200)
    cv2.moveWindow('image2', 600, 200)

    while True:
        k = cv2.waitKey(0)
        if k == ord('y') or k == ord('n') or k == 27: # 27 = escape
            cv2.destroyAllWindows()
            break

    return k


if __name__ == "__main__":
    #show_img("test1.jpeg", "test1.jpeg", "./")
    l = get_unjudged_list("rawlist.txt", "labeledlist.txt")
    do_label(l, "labeledlist.txt")

