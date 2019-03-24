import urllib
import cv2
import os
import numpy as np
from multiprocessing.dummy import Pool as ThreadPool 
import itertools

pic_num = 1

def store_raw_images(folders, links):
    pic_num = 1
    for link, folder in zip(links, folders):
        if not os.path.exists(folder):
            os.makedirs(folder)
        image_urls = str(urllib.urlopen(link).read())
        
        for i in image_urls.split('\\n'):
            try:                
                urllib.urlretrieve(i, folder + "/"+str(pic_num) + ".jpg")
                img = cv2.imread(folder + "/" + str(pic_num) + ".jpg")                         
                
                # Do preprocessing if you want
                if img is not None:
                    cv2.imwrite(folder + "/" + str(pic_num) + ".jpg", img)
                    pic_num += 1

            except Exception as e:
                    print(str(e))  
    
def removeInvalid(dirPaths):
    for dirPath in dirPaths:
        for img in os.listdir(dirPath):
            for invalid in os.listdir('invalid'):
                try:
                    current_image_path = str(dirPath) + '/' + str(img)
                    invalid = cv2.imread('invalid/' + str(invalid))
                    question = cv2.imread(current_image_path)
                    if invalid.shape == question.shape and not(np.bitwise_xor(invalid,question).any()):
                        os.remove(current_image_path)
                        break

                except Exception as e:
                    print(str(e))
  
def main():
    links = [ 
            'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n01318894', \
            'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n03405725', \
            'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07942152', \
            'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n00021265', \
            'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07690019', \
            'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07865105', \
            'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07697537', \
            'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n07690019']
    
    paths = ['pets', 'furniture', 'people', 'food', 'frankfurter', 'chili-dog', 'hotdog', 'hotdog bun']
    
    #store_raw_images(paths, links)
    removeInvalid('hotdog')
    removeInvalid('not hotdog')

if __name__ == "__main__":

    main()