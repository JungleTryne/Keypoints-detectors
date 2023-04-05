import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def match_corners(corners1, corners2, threshold=10):
    matched_pairs = []

    for i, c1 in enumerate(corners1):
        c1_tiled = np.tile(c1, (len(corners2), 1))
        distances = np.linalg.norm(c1_tiled - corners2, axis=1)
        min_distance_index = np.argmin(distances)

        if distances[min_distance_index] < threshold:
            matched_pairs.append((i, min_distance_index))

    return matched_pairs


def get_image_pair_repeatability(image1, image2, algorithm):
    st = time.time()
    corners1, descr1 = algorithm(image1)
    corners2, descr2 = algorithm(image2)

    if descr1 is not None and descr2 is not None:
        matcher = cv2.BFMatcher(normType=cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descr1, descr2)
        matches = sorted(matches, key=lambda x: x.distance)

        thres = 0
        for i, x in enumerate(matches):
            thres = i

        et = time.time()
        return thres / len(corners1), (et - st) / len(corners1)

    else:
        corners1 = np.unique(corners1, axis=0)
        corners2 = np.unique(corners2, axis=0)

        matched_corners = match_corners(corners1, corners2, 10)
        first, _ = zip(*matched_corners)

        et = time.time()
        return len(first) / len(corners1), (et - st) / len(corners1)


def get_image_repeatability(image_num: int, algorithm):
    avg = 0
    avg_time = 0
    for i in range(1, 13):
        if i != image_num:
            image1 = cv2.imread(f'photos/{image_num}.tif')
            image2 = cv2.imread(f'photos/{i}.tif')
            avg_delta, avg_time_delta = get_image_pair_repeatability(image1, image2, algorithm)
            avg += avg_delta
            avg_time += avg_time_delta

    return avg / 11, avg_time / 11


def harris_corner_detection(image, k=0.04, threshold=0.1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, 2, 3, k)
    corners = np.argwhere(dst > threshold * dst.max())

    return corners, None


def sift_key_points(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray).astype('uint8')

    sift = cv2.SIFT_create()
    kp, descr = sift.detectAndCompute(gray, None)

    kp = np.array([point.pt for point in kp])
    return kp, None


def brisk_key_points(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray).astype('uint8')

    brisk = cv2.BRISK_create()
    kp, descr = brisk.detectAndCompute(gray, None)

    kp = np.array([point.pt for point in kp])
    return kp, None


def main():
    algorithms = [
        (harris_corner_detection, "Harris"),
        (sift_key_points, "SIFT"),
        (brisk_key_points, "BRISK"),
    ]

    for func, name in algorithms:
        time = 0
        xs = [i for i in range(1, 13)]
        ys = []
        for i, x in enumerate(xs):
            y, time_delta = get_image_repeatability(x, func)
            ys.append(y)
            time += time_delta

        time /= 12
        print(f"Avg time for 1 point  {time * 1000} ms")
        plt.title(name)
        plt.scatter(xs, ys)
        plt.show()


if __name__ == '__main__':
    main()
