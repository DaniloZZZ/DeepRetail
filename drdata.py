import os
import cv2
import numpy as np
from datetime import date
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
DATAPATH = './data'


def save_train_img(className, setName, frameId, image):
    d = DATAPATH+'/train/'+className+'/'
    n = setName+"_id"+frameId+(date.today()).strftime('_at_%d-0%m')+".drtri.jpg"
    try:
        os.stat(d)
    except:
        os.mkdir(d,0o777)
        print "Created new directory %s for class %s!"%(d,className)
    f= open(d+n,"w+")
    f.write(image)
    f.close()
    print "written %s file"%(d+n)

def get_train_img():
    # get list of classnames=folders names in train folder
    tdir = DATAPATH+ "/train"
    classList = [name for name in os.listdir(tdir) if os.path.isdir(tdir+"/"+name)]
    
    print "found classnames:%s"%classList
    #load every image in dir, put in array
    X  = []
    y = []
    clid = 0
    for cname in classList:
        cdir = tdir+"/"+cname+"/" 
        iplist = [cdir+name for name in os.listdir(cdir) ]
        print "found %d images for class %s"%(len(iplist),cname)
        # TODO: check file format
        for ip in iplist:
            # read the file. guaranteed to exist
            img = cv2.imread(ip)
            resized_image = cv2.resize(img, (299,299)) 
            X.append(resized_image)
            y.append(clid)
        clid = clid+1
    d =  DRData(X,y,classList)
    d.print_stat()
    return d

class DRData:
    def __init__(self,X,y,classnmes):
        self.labels = np.array(y)
        self.clsNames = classnmes

        self.X = np.array(X)
        lb = LabelBinarizer()
        self.y= lb.fit_transform(self.labels)
        self._lb= lb

        self.len = self.X.shape[0]
        self.h = self.X.shape[1]
        self.w = self.X.shape[2]
        self.ch = self.X.shape[3] 
        self.cls = len(self._lb.classes_)

    def print_stat(self):
        print "X shape:%s \ny shape:%s \nclass names: %s \n"%\
        (self.X.shape,self.y.shape,self.clsNames)

    def r_s_split(self,ratio=  0.2):
        X_tr,X_s,y_tr,y_s = train_test_split(self.X,self.y,
                test_size=ratio,
                random_state=41)
        self.trn = (X_tr, y_tr)
        self.tst = (X_s, y_s)
        return X_tr,y_tr,X_s,y_s

def get_unique_labels_cnt(y):
    return len(set(y))
    

