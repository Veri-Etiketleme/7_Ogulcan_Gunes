import os
import imagehash
from PIL import Image


def search_images(path, subdirs):
    images_list = []

    print("Scanning files..")

    if subdirs == True:
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(".jpg" or ".jpeg" or ".png"):
                    images_list.append(os.path.join(root, file))
    else:
        for file in os.listdir(path):
            if file.endswith(".jpg" or ".jpeg" or ".png"):
                images_list.append(os.path.join(path, file))

    return images_list


def hash_image(image):
    hash = imagehash.average_hash(Image.open(image))
    return hash


def find_similar(hashes_cache, threshold, index):

    for y in range(index + 1, len(hashes_cache)):

        if hashes_cache[index] - hashes_cache[y] < threshold:
            return (index, y)