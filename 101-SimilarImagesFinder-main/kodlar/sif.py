from functools import partial
from multiprocessing import Pool
import utils
import time
from tqdm import tqdm
import argparse


images_list = []
similar_list = []
hashes_cache = []


def exec_hashes_cache_process(pool, n_iterations):
    print("Caching hashes..")
    return list(tqdm(pool.imap(utils.hash_image, images_list), total=n_iterations))


def exec_find_similar_process(pool, n_iterations, hashes_cache, threshold):
    print("Searching similar images..")
    find_similar_func = partial(utils.find_similar, hashes_cache, threshold)
    return [x for x in list(tqdm(pool.imap(find_similar_func, range(0, n_iterations)), total=n_iterations)) if x is not None]


def main(processes, threshold):

    pool = Pool(processes=processes)
    n_iterations = len(images_list)

    hashes_cache = exec_hashes_cache_process(pool, n_iterations)
    similar_list = exec_find_similar_process(pool, n_iterations, hashes_cache, threshold)

    return similar_list


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("-subdirs", action="store_true")
    parser.add_argument("-processes", nargs='?', const=1, default=1, type=int, help="Number of CPUs to use.")
    parser.add_argument("-threshold", nargs='?', const=4, default=4, type=int, help="The similarity treshold when comparing two images.")

    return parser.parse_args()


if __name__ == '__main__':
    start_time = time.time()

    args = parser()

    images_list = utils.search_images(args.path, args.subdirs)
    similar_list = main(args.processes, args.threshold)

    if len(similar_list) <= 0:
        print("\nNo similar images found.")
    else:
        for item in similar_list:
            print("\n======================")
            print(images_list[item[0]])
            print(images_list[item[1]])


    print("\n\nElapsed time: %s seconds" % (time.time() - start_time), "\n")
