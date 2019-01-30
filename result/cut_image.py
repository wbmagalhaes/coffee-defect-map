from PIL import Image
import os
import glob


def crop(img_path, cols, rows):
    img = Image.open(img_path)

    img_width, img_height = img.size
    crop_width = int(img_width / cols)
    crop_height = int(img_height / rows)

    for i in range(cols):
        for j in range(rows):
            box = (i * crop_width, j * crop_height, (i + 1)
                   * crop_width, (j + 1) * crop_height)

            cropped_img = img.crop(box)

            filename = os.path.splitext(img_path)[0]
            new_path = filename + '_cut_%s_%s' % (i, j) + ".jpg"

            cropped_img.save(new_path)


for addr in glob.glob('result/*.jpg'):
    crop(addr, 2, 3)
