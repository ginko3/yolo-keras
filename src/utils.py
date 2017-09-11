import numpy as np

class BoundBox:

    def __init__(self, n_class, x=0., y=0., w=0., h=0., c=0., index_label=-1):
        self.xmin, self.ymin, self.xmax, self.ymax, self.c = x, y, w, h, c
        self.probs = np.zeros((n_class,))
        self.index_label = index_label


    def compute_iou(self, box):
        # determine the (x, y)-coordinates of the intersection rectangle
        xA = max(self.xmin, box.xmin)
        yA = max(self.ymin, box.ymin)
        xB = min(self.xmax, box.xmax)
        yB = min(self.ymax, box.ymax)

        if xA < xB and yA < yB:
            # compute the area of intersection rectangle
            interArea = (xB - xA) * (yB - yA)
            # compute the area of both the prediction and ground-truth
            # rectangles
            boxAArea = (self.xmax - self.xmin) * (self.ymax - self.ymin)
            boxBArea = (box.xmax - box.xmin) * (box.ymax - box.ymin)
            # compute the intersection over union by taking the intersection
            # area and dividing it by the sum of prediction + ground-truth
            # areas - the intersection area
            iou = interArea / float(boxAArea + boxBArea - interArea)
        else:
            iou = 0

        assert iou >= 0
        assert iou <= 1.01

        return iou

def sigmoid(x):
    return 1. / (1. + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)
