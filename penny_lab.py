import csv
import pickle
import sys
from getopt import getopt, GetoptError

import cv2
import numpy as np

PIXEL_CACHE_FILE = 'pixel_cache.pickle'
EXPORT_FILE = 'colors.csv'


def main():
    # read in CLI args
    verbose = False
    scaling_factor = .5
    visual_output = False
    try:
        opts, args = getopt(sys.argv[4:], "hvc")
    except GetoptError:
        display_help()
        raise ValueError("Bad option flags.")
    for opt, arg in opts:
        if opt == '-h':
            display_help()
            sys.exit(0)
        elif opt == '-v':  # verbose
            verbose = True
        elif opt == '-s':
            scaling_factor = float(arg)
        elif opt == '-c':
            visual_output = True

    if len(sys.argv) < 3:
        display_help()
        raise ValueError("Not enough arguments.")

    image_path = sys.argv[1]
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError("No image found at: {0}. Exiting.".format(image_path))

    if verbose:
        print("Image found at {0} is {1} by {2}\n\n.".format(image_path, image.shape[1], image.shape[0]))

    min_dist = sys.argv[2]
    min_radius = sys.argv[3]

    # Perform edge detection
    print("Detecting pennies...")
    circles = find_circles(image, min_dist, min_radius)
    circles = np.round(circles[0, :]).astype("int")
    if circles is None or len(circles) == 0:
        print("No pennies found. Exiting.\n\n")
        exit(1)

    print("Found {0} pennies.\n\n".format(len(circles)))
    if verbose:
        circles = np.round(circles[0, :]).astype("int")
        for i, (x, y, r) in enumerate(circles):
            print("Penny #{0}: (x, y) = ({1}, {2}), r = {3}.".format(i, x, y, r))
        print("\n\n")

    # Display results 
    circles, scaled_image = scale(circles, image, scaling_factor)
    colors = analyze_color(circles, scaled_image, verbose)
    print("Writing colors in xyrRGB format to {0}...".format(EXPORT_FILE))
    with open('%s' % EXPORT_FILE, 'w+') as f:
        writer = csv.writer(f)
        writer.writerows(colors)
    if visual_output:
        show_color_circles(circles, image, colors, scaling_factor)


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


def show_color_circles(circles, image, colors, scaling_factor, image_duration=0):
    for (x, y, radius, r, g, b) in colors:
        # draw the circle itself
        x = int(x / scaling_factor)
        y = int(y / scaling_factor)
        radius = int(radius / scaling_factor)
        r = int(r)
        g = int(g)
        b = int(b)
        cv2.circle(image, (x, y), radius, (b, g, r), -1)

    cv2.imshow("image", image)
    cv2.waitKey(image_duration)
    pass


def analyze_color(circles, image, verbose):
    pixel_cache = load_pixel_cache()

    image_height = image.shape[0]
    image_width = image.shape[1]

    num_circles = len(circles)
    colors = np.zeros(shape=(num_circles, 6))
    for i, (x_center, y_center, radius) in enumerate(circles):
        if verbose:
            print("Processing penny #{0}/{1} located at ({2}, {3}) with radius = {4}..."
                  .format(i + 1, num_circles, x_center, y_center, radius))
        else:
            print("Processing penny #{0}/{1}".format(i + 1, num_circles))
        pixels = get_pixels(x_center, y_center, radius, pixel_cache)
        save_pixel_cache(pixel_cache)

        color_sum = np.zeros(3)
        for (x_pixel, y_pixel) in pixels:
            if y_pixel < image_height and x_pixel < image_width:
                color = image[y_pixel][x_pixel]
                color_sum = np.add(color, color_sum)

        colors[i] = np.concatenate((np.array([x_center, y_center, radius]),
                                    bgr_to_rgb(np.divide(color_sum, len(pixels)))))
        if verbose:
            print("Average color for penny #{0}: {1}\n".format(i + 1, ', '.join(colors[i].astype(str))))

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
        with open(PIXEL_CACHE_FILE, 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        open(PIXEL_CACHE_FILE, 'x')
        return {}


def save_pixel_cache(pixel_cache):
    with open(PIXEL_CACHE_FILE, 'wb') as f:
        pickle.dump(pixel_cache, f)


def get_pixel_cache_key(c_x, c_y, r):
    return "x:{0}_y:{1}_r:{2}".format(c_x, c_y, r)


# Utility functions
def scale(circles, image, scaling_factor):
    print("Scaling by {0}.".format(scaling_factor))
    circles = scale_circles(circles, scaling_factor)
    image = scale_image(image, scaling_factor)
    return circles, image


def scale_image(image, scaling_factor):
    return cv2.resize(image, (0, 0), fx=scaling_factor, fy=scaling_factor)


def scale_circles(circles, scaling_factor):
    return np.multiply(circles.copy(), scaling_factor)


def bgr_to_rgb(color):
    new_color = color.copy()
    new_color[2] = color[0]
    new_color[0] = color[2]
    return new_color


# UI
def display_help():
    print("Usage:\n")
    print("penny_lab.py [image_path] [min_dist] [min_radius] [show_circles]\n\n")

    print("Parameters:\n")
    print("min_dist: minimum distance between two circles in shape detection, measured in pixels\n")
    print("min_radius: minimum radius of a circle in shape detection, measured in pixels\n")
    print("show_circles: whether an image of the circles drawn over the original image is shown\nn")

    print("Optional flags:\n")
    print("-s: manually specify the scaling factor that is applied between edge detection and color averaging. "
          "Defaults to 0.5\n")
    print("-v: verbose mode, print more info\n")
    print("-h: show this help information\n")


if __name__ == "__main__":
    main()
