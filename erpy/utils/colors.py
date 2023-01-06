import numpy as np
from PIL import ImageColor

hex_green = "#7db5a8"
hex_bright_green = "#83f28f"
hex_red = "#b75659"
hex_orange = "#d6ae72"
hex_gray = "#595959"
hex_blue = "#73c2fb"

rgba_green = np.array(ImageColor.getcolor(hex_green, "RGBA")) / 255
rgba_bright_green = np.array(ImageColor.getcolor(hex_bright_green, "RGBA")) / 255
rgba_red = np.array(ImageColor.getcolor(hex_red, "RGBA")) / 255
rgba_orange = np.array(ImageColor.getcolor(hex_orange, "RGBA")) / 255
rgba_gray = np.array(ImageColor.getcolor(hex_gray, "RGBA")) / 255
rgba_blue = np.array(ImageColor.getcolor(hex_blue, "RGBA")) / 255

print(f"Green:  {rgba_green * 255}")
print(f"Orange: {rgba_orange * 255}")
print(f"Red:    {rgba_red * 255}")