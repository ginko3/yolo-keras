import numpy as np
import cv2

def display_output(output_it, draw=True, video=False):
    for frame, boxes in output_it:
        if draw:
            for xmin, ymin, xmax, ymax, label, c in boxes:
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)
                label_text = "{} {}%".format(label, c)
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0]

                # Create a colored rectangle behind the text
                cv2.rectangle(frame, (xmin-2, ymin-label_size[1]), (xmin+label_size[0], ymin), (255, 0, 0), thickness=-1)
                cv2.putText(frame, label_text, (xmin, ymin), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), thickness=1)

        if video:
            _display_video(frame, output_it)
        else:
            _display_image(frame, output_it)

def write_output(output_it):
    index_frame = 0
    for frame, boxes in output_it:
        for xmin, ymin, xmax, ymax, label, probability in boxes:
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite("images/out/{}.jpg".format(index_frame), frame)
        index_frame += 1


def _display_video(frame, output_it):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("yolo detection", frame)
    if cv2.waitKey(1) == ord("q"):
        output_it.send(False)
def _display_image(frame, output_it):

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("yolo detection", frame)
    if cv2.waitKey(0) == ord("q"):
        output_it.send(False)
