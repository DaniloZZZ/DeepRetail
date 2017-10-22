
def load_images(vid_paths):
    res = []
    for p in vid_paths:
        try:
            f= open('path')
            vid = f.read(f)
            images = vid_to_img(vid)
            print "converted video ",p," to ",len(images)
            res.append(images)
            f.close()
        except Exeption: 
            print Exception
    return res

def load_images(vid):
    fo