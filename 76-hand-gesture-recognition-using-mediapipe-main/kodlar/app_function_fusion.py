#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import mediapipe as mp

from utils.func import *
import pymouse
import pykeyboard
import time

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

from collections import Counter
from collections import deque
from enum import Enum


class State(Enum):
    Reset = 0
    Pointer = 1
    Scroll = 2
    Zoom = 3
    Tab = 4
    Shut = 5
    Shuted = 6
    Lock = 7
    Drag = 8


def first(hand_sign, hand_side, m, hold_time, tposition, pre_time, point_info, press, mouse_status, click_time,
          lr_state):
    # print(point_info)
    x1 = int(point_info[0] * point_info[2] / point_info[1])  # 0:pointx 1:cap_width 2:win_width
    y1 = int(point_info[3] * point_info[5] / point_info[4])  # 3:pointy 4:cap_height 5:win_height
    # x1 = int((point_info[0] - 0.1 * point_info[1]) / point_info[1] * point_info[2] * (5 / 3))
    # y1 = int((point_info[3] - 0.1 * point_info[4]) / point_info[4] * point_info[5] * (2))
    bias = (x1 - tposition[0]) ** 2 + (y1 - tposition[1]) ** 2
    tposition = [x1, y1]
    origin_mouse_status = mouse_status
    action_count = 20
    if bias > 200:
        hold_time = 0
        pre_time = time.time()
        press = 0
    else:
        diff = time.time() - pre_time
        hold_time = hold_time + diff
        pre_time = time.time()

    if mouse_status != 'drag' and lr_state[0] == 'Close' and lr_state[1] == 'Pointer':
        x = m.position()[0]
        y = m.position()[1]
        m.press(x, y)
        mouse_status = 'drag'
    if mouse_status == 'drag' and lr_state[0] == 'Open' and lr_state[1] == 'Pointer':
        x = m.position()[0]
        y = m.position()[1]
        m.release(x, y)
        mouse_status = 'None'

    if hand_info == 'Right':
        if lr_state[1] == 'Pointer' and bias > 15:
            m.move(x1, y1)
            if mouse_status != 'drag' and state_count[1] > action_count:
                state_count[1] = 0
                m.click(x1, y1, 1, 1)
                # press = 1
                mouse_status = 'click'
        if lr_state[1] == 'Two' and state_count[1] > action_count:
            state_count[1] = 0
            x = m.position()[0]
            y = m.position()[1]
            m.click(x, y, 1, 2)
            # press = 1
            mouse_status = 'double click'

        if lr_state[1] == 'Seven' and state_count[1] > action_count:
            state_count[1] = 0
            x = m.position()[0]
            y = m.position()[1]
            m.click(x, y, 2, 1)
            mouse_status = 'right click'
        # if hand_sign == 'Open' and hand_side == 'Right':
        #     x = m.position()[0]
        #     y = m.position()[1]
        #     m.release(x, y)
        #     mouse_status = 'Pointer'
        # origin_mouse_status = 'Pointer'
    # if origin_mouse_status == 'drag':
    #     mouse_status = 'drag'
    return hold_time, tposition, pre_time, press, mouse_status, click_time


def screen_mapping(cap_width, cap_height, win_width, win_height):
    if (cap_width * win_height) / (cap_height * win_width) < 0.9:
        cap_height = (cap_width * win_height) / win_width
    else:
        cap_height = cap_height * 0.9
        cap_width = ((cap_height * win_width) / win_height)
    return cap_height, cap_width


# deprecated
def control_keypoints(sign, hand, hand_pos):
    global now
    global still_t
    if hand == 'Right':
        duration = time.time() - now
        x = int(hand_pos[0] / cap_width * win_width)
        y = int(hand_pos[1] / cap_height * win_height)

        if sign == 'Pointer' and duration > 0.05:
            pre_pos = m.position()
            now = time.time()
            m.move(x, y)
            # print("move mouse to ", x, y)

            if dist(pre_pos, [x, y]) < 5:
                # print(dist(pre_pos, [x, y]))
                still_t = still_t + duration
            else:
                still_t = 0
            if still_t > 0.8:
                # m.press(x, y)
                still_t = 0
                print("press mouse at ", x, y)
        # elif sign == 'Open':
        #     m.release(x, y)

    elif hand == 'Left':
        pass
        if sign == 'Close':
            x = m.position()[0]
            y = m.position()[1]
            m.click(x, y)
            print("click at ", x, y)


# deprecated
def contorl_action(shut_history):
    # print(shut_history)
    if len(shut_history) == 2 and dist(shut_history[0], shut_history[1]) > 600 \
            and shut_history[1][0] > shut_history[0][0]:
        print(dist(shut_history[0], shut_history[1]))
        print("shutdown\n")
        shut_history.clear()


# 检测是否被锁
def check_lock(action_info, count):
    global state
    if action_info != 'Clockwise' and action_info != 'Counter Clockwise' \
            or (state != State.Pointer and state != State.Lock):
        return 0

    count += 1
    if count > 20:
        if state == State.Lock and action_info == 'Clockwise':
            state = State.Reset
            print("Unlock Successfully!!!\n")
        if state != State.Lock and action_info == 'Counter Clockwise':
            state = State.Lock
            print("Lock NOW!!!\nClockwise to unlock!\n")
        return 0
    return count


# 左/右手、手势、动作
def detect_state(hand_info, hand_sign, action_info, lr_state):  # 黄强补充好scroll和zoom的状态判断
    global state
    global m
    global k
    if lr_state[1] == 'Open':
        state = State.Reset
        # m.release(m.position()[0], m.position()[1])
        k.release_key(k.alt_key)
        # k.tap_key(k.escape_key)
    if lr_state[1] == 'Three':
        if state != State.Tab:
            state = State.Tab
            k.press_key(k.alt_key)
    else:
        k.release_key(k.alt_key)
    if lr_state[1] in ['Pointer', 'Two', 'Seven']:
        state = State.Pointer
    # if lr_state[1] == 'Two':
    #     state = State.Pointer
    # if lr_state[1] == 'Seven':
    #     state = State.Pointer
    if lr_state[1] == 'Six':
        state = State.Shut
    elif state == State.Shut and (lr_state[1] == 'Love' or lr_state[1] == 'Close') and state_count[1] > 20:
        state = State.Shuted
        k.press_key(k.alt_key)
        k.tap_key(k.function_keys[4])
        k.release_key(k.alt_key)
        print(time.time(), 'Shut Down')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)

    args = parser.parse_args()

    return args


# Global params
args = get_args()

cap_device = args.device
cap_width = args.width
cap_height = args.height

use_static_image_mode = args.use_static_image_mode
min_detection_confidence = args.min_detection_confidence
min_tracking_confidence = args.min_tracking_confidence

m = pymouse.PyMouse()
k = pykeyboard.PyKeyboard()

win_width = m.screen_size()[0]
win_height = m.screen_size()[1]
now = time.time()
still_t = 0
lock_state = True
state = State.Lock
last_active_time = time.time()
state_count = [0, 0]

if __name__ == '__main__':

    use_brect = True
    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)
    # モデルロード #############################################################
    mp_hands = mp.solutions.hands
    hand1 = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    hand2 = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    hands = [hand1, hand2]

    keypoint_classifier = KeyPointClassifier()
    point_history_classifier = PointHistoryClassifier()

    # ラベル読み込み ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # 座標履歴 #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)
    shut_history = deque(maxlen=2)
    tab_history = deque(maxlen=2)
    # フィンガージェスチャー履歴 ################################################
    finger_gesture_history = deque(maxlen=history_length)
    #  ########################################################################
    mode = 0

    # Custom Variable
    # m1 = pymouse.PyMouse()
    win_width = m.screen_size()[0]
    win_height = m.screen_size()[1]
    hold_time = 0
    now = time.time()
    tposition = [0, 0]
    press = 0
    pre_time = time.time()
    mouse_status = 'Pointer'
    # print(cap_width,cap_height)
    cap_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    cap_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    # print(cap_width, cap_height)
    new_cap_height, new_cap_width = screen_mapping(cap_width=cap_width, cap_height=cap_height, win_width=win_width,
                                                   win_height=win_height)
    click_time = time.time()
    tab_time = time.time()
    lock_count = 0  # 锁定状态数
    lr_state = ['None', 'None']
    hand_mode = 0  # 控制单手模式还是双手模式
    while True:
        # 获取图像
        fps = cvFpsCalc.get()
        key = cv.waitKey(10)
        if key == 0:  # End
            break
        ret, image = cap.read()
        if not ret:
            print('No camera detected!')
            break
        image = cv.flip(image, 1)  # ミラー表示
        debug_image = copy.deepcopy(image)

        # 検出実施 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False

        if lr_state[0] == 'Open' and state_count[0] > 50 and hand_mode == 0:
            hand_mode = 1
            print('Switch to TwoHands mode')
        elif lr_state[0] == 'None' and lr_state[1] == 'Open' and state_count[1] > 50 and hand_mode == 1:
            hand_mode = 0
            print('Switch to OneHand mode')
        results = hands[hand_mode].process(image)

        image.flags.writeable = True
        number, mode = select_mode(key, mode)

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # 外接矩形の計算
                brect = calc_bounding_rect(debug_image, hand_landmarks)
                # ランドマークの計算
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # 相対座標・正規化座標への変換
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # 学習データ保存
                logging_csv(number, mode, pre_processed_landmark_list,
                            pre_processed_point_history_list)

                # ハンドサイン分類
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # 指差しサイン
                    point_history.append(landmark_list[8])  # 人差指座標
                # elif hand_sign_id == 6:  # 手势为4（关闭程序指令）
                #     shut_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])
                    shut_history.append([0, 0])

                # フィンガージェスチャー分類
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # 直近検出の中で最多のジェスチャーIDを算出
                finger_gesture_history.append(finger_gesture_id)
                most_common_fg_id = Counter(
                    finger_gesture_history).most_common()

                # 描画
                debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                debug_image = draw_landmarks(debug_image, landmark_list)
                debug_image = draw_info_text(
                    debug_image,
                    brect,
                    handedness,
                    keypoint_classifier_labels[hand_sign_id],
                    point_history_classifier_labels[most_common_fg_id[0][0]],
                )

                hand_num = len(results.multi_hand_landmarks)  # 手的个数
                hand_info = handedness.classification[0].label[0:]  # 左右手信息
                hand_sign = keypoint_classifier_labels[hand_sign_id]  # 手势
                action_info = point_history_classifier_labels[most_common_fg_id[0][0]]  # 动作
                pointer_pos = landmark_list[8]  # pointer坐标
                # 保存左右手状态信息和持续时长
                if hand_info == 'Left':
                    if lr_state[0] == hand_sign:
                        state_count[0] += 1
                    else:
                        state_count[0] = 0
                        lr_state[0] = hand_sign
                    if hand_num == 1:
                        lr_state[1] = 'None'
                elif hand_info == 'Right':
                    if lr_state[1] == hand_sign:
                        state_count[1] += 1
                    else:
                        state_count[1] = 0
                        lr_state[1] = hand_sign
                    if hand_num == 1:
                        lr_state[0] = 'None'
                print(lr_state, state_count)
                # print(lr_state)
                # print("INFO:", landmark_list[8], hand_info)
                # print(keypoint_classifier_labels[hand_sign_id], landmark_list[8])
                # print(shut_history)
                lock_count = check_lock(action_info, lock_count)
                if state != State.Lock:
                    detect_state(hand_info, hand_sign, action_info, lr_state)  # 检测当前状态
                    if state == State.Pointer:
                        hand_info = handedness.classification[0].label[0:]
                        point_info = []
                        point_info.append(landmark_list[8][0])
                        point_info.append(new_cap_width)
                        point_info.append(win_width)
                        point_info.append(landmark_list[8][1])
                        point_info.append(new_cap_height)
                        point_info.append(win_height)
                        hold_time, tposition, pre_time, press, mouse_status, click_time = first(
                            hand_sign=keypoint_classifier_labels[hand_sign_id], hand_side=hand_info, m=m,
                            hold_time=hold_time,
                            tposition=tposition, pre_time=pre_time, point_info=point_info, press=press,
                            mouse_status=mouse_status, click_time=click_time, lr_state=lr_state)
                    if state == State.Scroll:
                        pass  # 黄强补充
                    if state == State.Zoom:
                        pass  # 黄强补充
                    if state == State.Tab:
                        if time.time() - tab_time > 0.5:
                            k.tap_key(k.tab_key)
                            tab_time = time.time()
                    else:
                        k.release_key(k.alt_key)
                last_active_time = time.time()
        else:
            point_history.append([0, 0])
            lr_state = ['None', 'None']
            if state != State.Lock and time.time() - last_active_time > 30:
                # print(time.time() - last_active_time)
                print("No active action detected.\nLOCK NOW!!!\nClockwise to unlock!")
                state = State.Lock

        debug_image = draw_point_history(debug_image, point_history)
        debug_image = draw_info(debug_image, fps, mode, number, state, mouse_status, hand_mode)

        # 画面反映 #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()
