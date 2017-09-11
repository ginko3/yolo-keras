#!/usr/bin/env python3
# coding: utf-8

import json
import os
import argparse

from src.yoloNetwork import YoloNetwork
from src.iterators import frameIt, cameraIt
from src.visualization import display_output, write_output

parser = argparse.ArgumentParser("Runs YOLO from model.")

parser.add_argument("model_path", help="path to model.h5", default="model/yolo-person.h5")
parser.add_argument("config_path", help="path to config (json file)", default="cfg/yolo-person.json")
parser.add_argument("-i", "--iterator", help="iterator type", default="image")
parser.add_argument("-n", "--noui", help="hides network output", action='store_true')

# TODO: Test on video
# TODO: add verbose option
# TODO: colors for classes

def main(model_params_path, model_network_path, frame_it, video=False, b_display=True, verbose=True):
    # Set tensorflow verbose
    if not verbose:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    # Loads model parameters from json
    with open(model_params_path, "r") as f:
        model_params = json.loads(f.read())

    yoloNet = YoloNetwork(model_network_path, model_params)

    output_it = yoloNet.detect(frame_it)

    if b_display:
        display_output(output_it, video=video)
    else:
        write_output(output_it)

if __name__ == '__main__':
    model_network_path = parser.parse_args().model_path
    model_params_path = parser.parse_args().config_path
    iterator_type = parser.parse_args().iterator
    b_display = not parser.parse_args().noui

    if iterator_type == "image":
        frame_it = frameIt()
        video = False
    elif iterator_type == "camera":
        frame_it = cameraIt()
        video = True
    else:
        raise NameError("Iterator type {} not understood.".format(iterator_type))

    main(model_params_path, model_network_path, frame_it, video=video, b_display=b_display, verbose=False)
