import os
from PIL import Image
from diffusers.utils.export_utils import export_to_gif


def export_to_gif_mv(samples, save_path):
    num_images = len(samples[0])
    widths, heights = samples.shape
    # 用于保存横向拼接后的图像的列表
    concat_images = []

    # 遍历每个图像索引
    for i in range(num_images):
        # 提取每个子列表中的第i个图像
        images_to_concat = samples[i]

        # 获取所有图像的宽度和高度
        # widths, heights = zip(*(i.size for i in images_to_concat))
        
        # 计算拼接后的总宽度和最大高度
        # total_width = sum(widths)
        # max_height = max(heights)
        
        # 创建一个新图像，它的宽度是所有图像宽度之和，高度是所有图像中的最大高度
        new_im = Image.new('RGB', (total_width, max_height))
        
        # 横向拼接图像
        x_offset = 0
        for im in images_to_concat:
            new_im.paste(im, (x_offset,0))
            x_offset += im.size[0]
        
        # 将拼接后的图像添加到结果列表中
        concat_images.append(new_im)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    export_to_gif(concat_images, save_path)
