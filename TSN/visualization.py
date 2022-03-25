import torch
from PIL import Image
import os
from utils import generate_label

def tensor_for_board(img_tensor):
    # print(img_tensor.shape)
    if img_tensor.size(1) == 1:
        tensor = img_tensor.repeat(1, 3, 1, 1)
        if torch.max(tensor <= 1.0):
            tensor = (tensor.clone() + 1) * 0.5
    elif img_tensor.size(1) > 3:
        tensor = generate_label(img_tensor, 256, 256)
    else:
        tensor = img_tensor
        if torch.max(tensor <= 1.0):
            tensor = (tensor.clone() + 1) * 0.5

    return tensor


def tensor_list_for_board(img_tensors_list):
    grid_h = len(img_tensors_list)
    grid_w = max(len(img_tensors) for img_tensors in img_tensors_list)

    batch_size, channel, height, width = tensor_for_board(img_tensors_list[0][0]).size()
    canvas_h = grid_h * height
    canvas_w = grid_w * width
    canvas = torch.FloatTensor(batch_size, channel, canvas_h, canvas_w).fill_(0.5)
    for i, img_tensors in enumerate(img_tensors_list):
        for j, img_tensor in enumerate(img_tensors):
            offset_h = i * height
            offset_w = j * width
            tensor = tensor_for_board(img_tensor)
            # print(tensor.shape)
            canvas[:, :, offset_h: offset_h + height, offset_w: offset_w + width].copy_(tensor)

    return canvas


def board_add_image(board, tag_name, img_tensor, step_count):
    tensor = tensor_for_board(img_tensor)

    for i, img in enumerate(tensor):
        board.add_image('%s/%03d' % (tag_name, i), img, step_count)


def board_add_images(board, tag_name, img_tensors_list, step_count):
    tensor = tensor_list_for_board(img_tensors_list)

    for i, img in enumerate(tensor):
        board.add_image('%s/%03d' % (tag_name, i), img, step_count)


def save_images(img_tensors, img_names, save_dir):
    for img_tensor, img_name in zip(img_tensors, img_names):
        tensor = (img_tensor.clone() + 1) * 0.5 * 255
        tensor = tensor.cpu().clamp(0, 255)

        array = tensor.numpy().astype('uint8')
        if array.shape[0] == 1:
            array = array.squeeze(0)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)

        Image.fromarray(array).save(os.path.join(save_dir, img_name))