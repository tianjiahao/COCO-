from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import time

time1= time.time()

GtFile = '/home/myubuntu/Desktop/openpose_train/training/dataset/COCO/annotations/person_keypoints_val2017.json'
imageDir = '/home/myubuntu/Videos/AlphaPose-pytorch -4yue/examples/res/val2017_out/val2017_out/vis'

coco_kps = COCO(GtFile)
catIds = coco_kps.getCatIds(catNms=['person'])
imgIds = coco_kps.getImgIds(catIds=catIds)
for i in imgIds:
    #i=30#np.random.randint(0, len(imgIds))
    #a=imgIds[i]
    img = coco_kps.loadImgs(i)[0]
    annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = coco_kps.loadAnns(annIds)
    plt.figure(i)
    I = io.imread('%s/%s' % (imageDir,img['file_name']))
    plt.imshow(I)
    plt.axis('off')
    coco_kps.showAnns(anns)
    #plt.show()
    plt.savefig('/home/myubuntu/Videos/AlphaPose-pytorch -4yue/examples/res/val2017_out/gt+pred/%s'% (i))
    plt.close()
time2=time.time()
print('spent  = %.2f min'%((time2-time1)/60))
print('well done!')