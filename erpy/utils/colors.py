import numpy as np
from PIL import ImageColor

hex_green = "#7db5a8"
hex_red = "#b75659"
hex_orange = "#d6ae72"
hex_gray = "#595959"

rgba_green = np.array(ImageColor.getcolor(hex_green, "RGBA")) / 255
rgba_red = np.array(ImageColor.getcolor(hex_red, "RGBA")) / 255
rgba_orange = np.array(ImageColor.getcolor(hex_orange, "RGBA")) / 255
rgba_gray = np.array(ImageColor.getcolor(hex_gray, "RGBA")) / 255
