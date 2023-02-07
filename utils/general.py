from PIL import Image


def image_padding(path, input_shape):
    img = Image.open(path)
    iw, ih = img.size
    h, w = input_shape

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2

    img = img.resize((nw, nh), Image.BICUBIC)
    new_img = Image.new('RGB', (w, h), (128, 128, 128))
    new_img.paste(img, (dx, dy))
    return new_img


if __name__ == '__main__':
    image_padding('/Users/icemoka/CodeField/Python/Blur/51-1/不可分类/0.299_100X_5924_1858.jpg', (200, 150))
