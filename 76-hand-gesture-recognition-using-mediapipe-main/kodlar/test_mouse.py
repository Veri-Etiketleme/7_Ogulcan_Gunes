import cv2
import pymouse
import PyHook3 as pyHook
import pythoncom
import time


def func():
    m = pymouse.PyMouse()  # 获取鼠标指针对象
    pos = m.position()
    # m.press(pos[0], pos[1])
    time.sleep(1)
    x = pos[0] + 50
    y = pos[1] + 50
    m.drag(x, y)  # 鼠标移动(x,y)坐标
    # m.release(x, y)
    print(pos)  # 获取当前鼠标指针的坐标



if __name__ == "__main__":
    func()
