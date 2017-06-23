import sys
import os

def pair_convetor(ifstr, ofstr, remove_header = False):
    of = open(ofstr,"w")

    path = "E:\work@bing\linkedin\img_data\images\\"

    with open(ifstr,"r") as f:
        line = None
        if remove_header:
            line = f.readline()

        for line in f:
            line = line.strip()
            cols  = line.split(",")
            if len(cols) != 5: continue
            #oline = ",".join([cols[0],path + cols[1]+"_face.jpg", path +cols[2]+"_face.jpg",cols[3],cols[4]])
            oline = ",".join([cols[0], cols[1] + "_face.jpg", cols[2] + "_face.jpg", cols[3], cols[4]])
            of.writelines(oline+"\n")
    of.close()


if __name__ == "__main__":
    print ("start converting...")
    pair_convetor("E:\work@bing\linkedin\img_data\data2.csv","rawlist.txt",True)


