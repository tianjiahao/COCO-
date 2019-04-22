from pycocotools.coco import COCO
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as io
import time

time1= time.time()
GtFile='/home/myubuntu/Desktop/标注coco/34-38/new.json'
cocoGt=COCO(GtFile)
DtFile = '/home/myubuntu/Videos/AlphaPose-pytorch/examples/res/50ren/alphapose-results.json'
cocoDt=cocoGt.loadRes(DtFile)
imageDir = '/home/myubuntu/Desktop/标注coco/34-38'

#coco_kps = COCO(annFile)
catIds = cocoDt.getCatIds(catNms=['person'])
imgIds = cocoDt.getImgIds(catIds=catIds)
for i in imgIds:
    #i=30#np.random.randint(0, len(imgIds))
    #a=imgIds[i]
    img = cocoDt.loadImgs(i)[0]
    annIds = cocoDt.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
    anns = cocoDt.loadAnns(annIds)
    plt.figure(i)
    I = io.imread('%s/%s' % (imageDir,img['file_name']))
    plt.imshow(I)
    plt.axis('off')
    cocoDt.showAnns(anns)
    #plt.show()
    plt.savefig('/home/myubuntu/Videos/AlphaPose-pytorch/examples/res/50ren/%s'% (i))
    plt.close()
time2=time.time()
print('spent time = %.2f min'%((time2-time1)/60))
print('well done!')


