import pickle


class Pixels:
    def __init__(self):
        self.PIXEL_CACHE_LOCATION = 'pixel_cache.pickle'

    def load_pixel_cache(self):
        try:
            with open(self.PIXEL_CACHE_LOCATION, self.rb) as f:
                return pickle.load(f)
        except FileNotFoundError:
            open(self.PIXEL_CACHE_LOCATION, 'x')
            return {}

    def save_pixel_cache(self, pixel_cache):
        with open(self.PIXEL_CACHE_LOCATION, 'wb') as f:
            pickle.dump(pixel_cache, f)

    def get_pixels(self, c_x, c_y, r, pixel_cache):
        key = self.get_pixel_cache_key(c_x, c_y, r)
        if key in pixel_cache:
            return pixel_cache[key]

        pixels = self.calculate_pixels(c_x, c_y, r)

        pixel_cache[key] = pixels
        return pixels

    @staticmethod
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

    @staticmethod
    def get_pixel_cache_key(c_x, c_y, r):
        return "x:{0}_y:{1}_r:{2}".format(c_x, c_y, r)
