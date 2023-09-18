import cv2
import numpy as np
import os
import csv
import sys
from utils import utils
import json

def loop(state, num_body_parts, training_data_paths):
    print("Current Trial Idx: " + str(state))
    global click_list
    global cur_num_clicks
    click_list = []
    cur_num_clicks = 0
    images = os.listdir(training_data_paths[state])
    utils.process_dir_list(images)
    image_paths = list(map(lambda x: os.path.join(training_data_paths[state], x), images))
    trial_data = []
    for image_path in image_paths:
        if image_path[-4:] != ".png": continue 
        img = cv2.imread(image_path)
        image_clicks = []
        sys.stdin.flush()
        while True:
            cv2.imshow('img', img)
            k = cv2.waitKey(1)
            if k == 127:
                if len(click_list) > 0:
                    del click_list[cur_num_clicks-1]
                    cur_num_clicks -= 1
                print("New List:")
                print(click_list)
                print_next()
            if k == ord('s'):
                print("Current List:")
                print(click_list)
                print_next()
            if k == 32:
                for j in range(0,num_body_parts):
                    click_list.append([-1,-1])
                cur_num_clicks += num_body_parts
                print(click_list)
                print_next()
            if k == 13:
                print("Saved List:")
                print(click_list)
                image_clicks = click_list.copy()
                click_list = []
                cur_num_clicks = 0
                print("New List:")
                print(click_list)
                break
        try:
            image_data = np.asarray(image_clicks,dtype=float).reshape(num_body_parts,2)
        except:
            raise ValueError('Your click list has incorrect length')
        trial_data.append(image_data)
    trial_data_np = np.concatenate(trial_data, axis=1).reshape(num_body_parts,8)
    out_path = os.path.join(training_data_paths[state], "labels.csv")
    if os.path.exists(out_path):
        os.remove(out_path)
    np.savetxt(out_path, trial_data_np, delimiter=',')
    state += 1
    print("Current Trial Idx: " + str(state))
    state_data = np.array([state])
    np.savetxt(state_tracker_path, state_data, delimiter=',')

if __name__ == "__main__":
    state_tracker_path = os.path.join(os.getcwd(), "labeling_state.csv")
    if not os.path.exists(state_tracker_path):
        state = np.array([0])
        np.savetxt(state_tracker_path, state, delimiter=',')
    state_start = int(np.genfromtxt(state_tracker_path, delimiter=','))
    training_data_dir = os.path.join(os.getcwd(), "Training_Data")
    training_data = os.listdir(training_data_dir)
    utils.process_dir_list(training_data)
    cv2.destroyAllWindows()
    sys.stdin.flush()

    training_data_paths = list(map(lambda x: os.path.join(training_data_dir, x), training_data))

    click_list = []
    cur_num_clicks = 0

    body_parts_path = os.path.join(os.getcwd(), "body_parts.json")
    
    body_parts = None
    with open(body_parts_path, 'r', encoding="utf-8") as f:
        body_parts = json.load(f)["names"]

    num_body_parts = len(body_parts)

    def print_next():
        global cur_num_clicks
        if cur_num_clicks > num_body_parts-1:
            print("List Full!")
        else:
            print("Next Body Part: " + body_parts[cur_num_clicks])
            

    def callback(event, x, y, flags, param):
        global cur_num_clicks
        global click_list
        if event == 1:
            print(body_parts[min(cur_num_clicks,num_body_parts-1)] + ": "+ str(x) + ", " + str(y))
            cur_num_clicks += 1
            click_list.append([x,y])
            print_next()
        elif event == 2:
            print("Recorded pass for "+ body_parts[min(cur_num_clicks,num_body_parts-1)] + ": -1, -1")
            cur_num_clicks += 1
            click_list.append([-1,-1])
            print_next()
        pass
    cv2.namedWindow('img')
    cv2.setMouseCallback('img', callback)

    state_start = int(np.genfromtxt(state_tracker_path, delimiter=','))

    state = state_start
    print(len(training_data))

    for i in range(state_start, len(training_data)):
        loop(i, num_body_parts, training_data_paths)
        


