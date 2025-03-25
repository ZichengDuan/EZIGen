import random
from PIL import Image, ImageOps

def random_horizontal_flip(image1: Image.Image, image2: Image.Image, p: float = 0.5) -> tuple[Image.Image, Image.Image]:
    """
    以概率 p 对 image1 和 image2 进行同步水平翻转。

    :param image1: 第一张 PIL 图像
    :param image2: 第二张 PIL 图像
    :param p: 翻转概率（默认 0.5）
    :return: 经过同步翻转后的两张图像
    """
    if random.random() < p:
        image1 = ImageOps.mirror(image1)  # 水平翻转
        image2 = ImageOps.mirror(image2)
    return image1, image2


def random_rotation(image: Image.Image, max_angle: float = 30) -> Image.Image:
    """
    对 image 进行随机旋转，角度范围为 ±max_angle，旋转后用白色填充以保持原尺寸。

    :param image: 输入 PIL 图像
    :param max_angle: 最大旋转角度（默认 ±30°）
    :return: 旋转后的图像（保持原尺寸）
    """
    angle = random.uniform(-max_angle, max_angle)  # 生成随机旋转角度
    return image.rotate(angle, resample=Image.BICUBIC, fillcolor=(255, 255, 255))


def random_scaling(image: Image.Image, scale_range: tuple[float, float] = (0.8, 1.2)) -> Image.Image:
    """
    对 image 进行随机缩放（±20%），缩放后用白色填充以保持原尺寸。

    :param image: 输入 PIL 图像
    :param scale_range: 缩放范围（默认 0.8 到 1.2）
    :return: 缩放后的图像（保持原尺寸）
    """
    w, h = image.size
    scale_factor = random.uniform(*scale_range)  # 生成随机缩放比例

    # 计算新的尺寸
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)

    # 进行缩放
    scaled_image = image.resize((new_w, new_h), resample=Image.BICUBIC)

    # 创建一个白色背景的原尺寸图片
    new_image = Image.new("RGB", (w, h), (255, 255, 255))

    # 计算粘贴位置（居中）
    paste_x = (w - new_w) // 2
    paste_y = (h - new_h) // 2

    # 粘贴缩放后的图像
    new_image.paste(scaled_image, (paste_x, paste_y))

    return new_image