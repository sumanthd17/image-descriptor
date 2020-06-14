import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools.coco import COCO

dataDir = "../data"
dataType = "val2014"
instanceFile = os.path.join(dataDir, "annotations/instances_{}.json".format(dataType))
annotationFile = os.path.join(dataDir, "annotations/captions_{}.json".format(dataType))

coco = COCO(instanceFile)
coco_captions = COCO(annotationFile)

ids = list(coco.anns.keys())
annotations_id = np.random.choice(ids)
image_id = coco.anns[annotations_id]["image_id"]
img = coco.loadImgs(image_id)[0]
url = img["coco_url"]

image = Image.open(
    os.path.join(
        dataDir, "val2014/COCO_{}_{}.jpg".format(dataType, str(image_id).zfill(12))
    )
)
plt.axis("off")
plt.imshow(image)
plt.show()

annIds = coco_captions.getAnnIds(imgIds=img["id"])
anns = coco_captions.loadAnns(annIds)
coco_captions.showAnns(anns)
