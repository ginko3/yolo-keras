import numpy as np
import cv2

from keras.models import load_model

from src.utils import BoundBox, sigmoid, softmax

class YoloNetwork:
    def __init__(self, model_network_path, model_params):
        self.model = load_model(model_network_path)

        self.handlesParam(model_params)

    def handlesParam(self, model_params):
        self.anchors = [float(anchors.strip()) for anchors in model_params["anchors"].split(',')]

        # Thresholds
        self.threshold = model_params["threshold"]
        self.threshold_iou = model_params["threshold_iou"]

        # Labels
        self.labels = model_params["labels"]


        # Network input shape
        if not "input_width" in model_params.keys() or not "input_height" in model_params.keys():
            _, self.input_width, self.input_height, self.input_channel = self.model.input_shape
        elif self.model.input_shape != (None, model_params["input_width"], model_params["input_height"], 3):
            raise ValueError("Expected input shape {}, got (None, {}, {}, 3).".format(self.model.input_shape, model_params["input_width"], model_params["input_height"]))
        else:
            self.input_width = model_params["input_width"]
            self.input_height = model_params["input_height"]

        # Yolo params
        if not "B" in model_params.keys():
            self.B = len(self.anchors) // 2
        elif len(self.anchors)/2 != model_params["B"]:
            raise ValueError("Got anchors for {} box, but B is set to {}.".format(len(self.anchors)//2, self.B))
        else:
            self.B = model_params["B"]

        if not "C" in model_params.keys():
            self.C = len(self.labels)
        elif len(self.labels) != model_params["C"]:
            raise ValueError("Got {} classes, but C is set to {}.".format(len(self.labels), self.C))
        else:
            self.C = model_params["C"]

        if not "S" in model_params.keys():
            self.S = self.model.output_shape[1]
        elif self.model.output_shape != (None, model_params["S"], model_params["S"], self.B*(5 + self.C)):
            raise ValueError("Expected output shape {0}, got (None, {1}, {1}, {2}).".format(self.model.output_shape, self.S, self.B*(5 + self.C)))
        else:
            self.S = model_params["S"]

    def detect(self, frame_it):
        for image in frame_it:
            # Resize image to network input
            image_input = cv2.resize(image, (self.input_width, self.input_height)) / 255.
            img_height, img_width, image_channels = image.shape

            # Convert to batch shape
            image_input = np.expand_dims(image_input, 0)

            netout = self.model.predict(image_input)[0]
            boxes = self.interpret_netout(netout, img_width, img_height)

            if (yield image, boxes) == False:
                return

    def interpret_netout(self, netout, image_width=608, image_height=608):
        # Reshape netout
        netout = netout.reshape((self.S, self.S, self.B, 5+self.C))

        # List of temporary boxes
        boxes = []

        # interpret the output by the network
        for row in range(self.S):
            for col in range(self.S):
                for b in range(self.B):
                    # First 5 are x, y, w, h and confidence
                    x, y, w, h, c = netout[row,col,b,:5]
                    # Last are class likelihoods
                    classes = netout[row,col,b,5:]

                    # Probability for each class
                    c = sigmoid(c)
                    classes = softmax(classes) * c
                    # Index of detected class
                    index_label = np.argmax(classes)

                    # Skip box if below threshold
                    if classes[index_label] < self.threshold:
                        continue

                    # Compute actual box position
                    x = (col + sigmoid(x)) / self.S
                    y = (row + sigmoid(y)) / self.S
                    w = self.anchors[2 * b + 0] * np.exp(w) / self.S
                    h = self.anchors[2 * b + 1] * np.exp(h) / self.S

                    # Create Box
                    box = BoundBox(self.C)
                    box.xmin  = int((x - w/2) * image_width)
                    box.ymin  = int((y - h/2) * image_height)
                    box.xmax  = int((x + w/2) * image_width)
                    box.ymax  = int((y + h/2) * image_height)

                    box.probs = classes
                    box.c = c
                    box.index_label = index_label

                    # Check for other box with high iou
                    addBox = True
                    for index_box, boxB in enumerate(boxes):
                        if box.index_label == boxB.index_label and box.compute_iou(boxB) > self.threshold_iou:
                            if box.probs[box.index_label] > boxB.probs[boxB.index_label]:
                                del boxes[index_box]
                            else:
                                addBox = False
                            break

                    if addBox:
                        boxes.append(box)

        for box in boxes:
            max_prob = int(box.probs[box.index_label] * 100)
            yield (box.xmin, box.ymin, box.xmax, box.ymax, self.labels[box.index_label], max_prob)
