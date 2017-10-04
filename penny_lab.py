import csv
import pickle
import sys

import cv2
import numpy as np

# penny_lab.py [image_path] [MIN_DIST_PENNIES] [MIN_RADIUS_PENNIES] [show_result]
PIXEL_CACHE_LOCATION = 'pixel_cache.pickle'


def bgr_to_rgb(color):
    new_color = color.copy()
    new_color[2] = color[0]
    new_color[0] = color[2]
    return new_color


def main():
    # read in CLI args
    image_path = sys.argv[1]
    image = cv2.imread(image_path)

    min_dist = sys.argv[2]
    min_radius = sys.argv[3]

    show_result = sys.argv[4] == '1'
    if image is None:
        raise FileNotFoundError("No image found at: {0}. Exiting.".format(image_path))

    print("Image is {0} by {1}\n\n.".format(image.shape[1], image.shape[0]))
    print("Detecting pennies...")
    circles = find_circles(image, min_dist, min_radius)
    if circles is None:
        print("No pennies found. Exiting.\n\n")
        exit(1)

    circles = np.round(circles[0, :]).astype("int")  # bad code
    print("Found {0} pennies.\n\n".format(len(circles)))
    for i, (x, y, r) in enumerate(circles):
        print("Penny #{0}: (x, y) = ({1}, {2}), r = {3}.".format(i, x, y, r))
    print("\n\n")

    if show_result:
        show_circles(circles, image, image_duration=5000)

    circles, image = scale(circles, image, .5)
    colors = analyze_color(circles, image)
    with open('colors.csv', 'w+') as f:
        writer = csv.writer(f)
        writer.writerows(colors)


def find_circles(image, min_dist, min_radius):
    """
    Uses cv2.HoughCircles to find center points and associated radii in an image.

    :rtype: float[]
    :param image: the image to find circles in
    :param min_dist: the minimum distance between two center points
    :param min_radius: the minimum radius of a circle
    :return:
    """

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # turn the image grayscale
    method = cv2.HOUGH_GRADIENT  # currently the only option
    dp = 2
    min_dist = int(min_dist)
    min_radius = int(min_radius)

    return cv2.HoughCircles(gray_image, method, dp, min_dist, minRadius=min_radius)


def show_circles(circles, image, image_duration=0):
    for (x, y, r) in circles:
        # draw the circle itself
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)
        # draw a 10x10 square at the center point
        cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

    cv2.imshow("image", image)
    cv2.waitKey(image_duration)


def analyze_color(circles, image):
    # set up pixel cache
    pixel_cache = load_pixel_cache()

    image_height = image.shape[0]
    image_width = image.shape[1]
    colors = np.zeros(shape=(len(circles), 3))
    for i, (x_center, y_center, radius) in enumerate(circles):
        print("Processing penny #{0} located at ({1}, {2}) with radius = {3}...".format(i, x_center, y_center, radius))

        pixels = get_pixels(x_center, y_center, radius, pixel_cache)
        save_pixel_cache(pixel_cache)

        color_sum = np.zeros(3)
        for (x_pixel, y_pixel) in pixels:
            if y_pixel < image_height and x_pixel < image_width:
                color = image[y_pixel][x_pixel]
                color_sum = np.add(color, color_sum)

        colors[i] = bgr_to_rgb(np.divide(color_sum, len(pixels)))

        print("Average color for penny #{0}: {1}\n".format(i, ', '.join(colors[i].astype(str))))

    return colors


# Pixels
def get_pixels(c_x, c_y, r, pixel_cache):
    key = get_pixel_cache_key(c_x, c_y, r)
    if key in pixel_cache:
        return pixel_cache[key]

    pixels = calculate_pixels(c_x, c_y, r)

    pixel_cache[key] = pixels
    return pixels


def calculate_pixels(c_x, c_y, r):
    c_x = int(c_x)
    c_y = int(c_y)
    r = int(r) * int(r)
    pixels = []
    for x in range(c_x - r, c_x + r):
        for y in range(c_y - r, c_y + r):
            dx = x - c_x
            dy = y - c_y
            r_d = np.square(dx) + np.square(dy)
            if r_d <= r:
                pixels.append([x, y])
    return pixels


# Pixel cache
def load_pixel_cache():
    try:
        with open(PIXEL_CACHE_LOCATION, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        open(PIXEL_CACHE_LOCATION, 'x')
        return {}


def save_pixel_cache(pixel_cache):
    with open(PIXEL_CACHE_LOCATION, 'wb') as f:
        pickle.dump(pixel_cache, f)


def get_pixel_cache_key(c_x, c_y, r):
    return "x:{0}_y:{1}_r:{2}".format(c_x, c_y, r)


# utility functions
def scale(circles, image, scaling_factor):
    print("Scaling by {0}.".format(scaling_factor))
    circles = scale_circles(circles, scaling_factor)
    image = scale_image(image, scaling_factor)
    return circles, image


def scale_image(image, scaling_factor):
    return cv2.resize(image, (0, 0), fx=scaling_factor, fy=scaling_factor)


def scale_circles(circles, scaling_factor):
    return np.multiply(circles.copy(), scaling_factor)


if __name__ == "__main__":
    main()
