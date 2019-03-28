#import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
# import numpy as np
# import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

annType = ['segm','bbox','keypoints']
annType = annType[2]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'

print ('Annotation Type: %s '%(annType))
#Running demo for *bbox* results.
#initialize COCO ground truth api      person_keypoints_val2017
annFile='/home/myubuntu/Videos/AlphaPose-pytorch-3yue/examples/new.json'
# dataType='val2017'
# annFile = '%s/annotations/new.json'%(dataDir)#,prefix,dataType)
cocoGt=COCO(annFile)

#initialize COCO detections api                kanshousuo_results.json    val2017_results.json
resFile='/home/myubuntu/Videos/AlphaPose-pytorch-3yue/examples/res/1/alphapose-results.json'
#resFile = resFile%(dataDir) #, prefix, dataType, annType)

cocoDt=cocoGt.loadRes(resFile)

imgIds=sorted(cocoGt.getImgIds())
#imgIds=imgIds[0:100]
#imgId = imgIds[np.random.randint(100)]

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()
