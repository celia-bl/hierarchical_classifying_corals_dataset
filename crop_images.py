import torch
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageDraw, ImageFont, ImageFile
from torch import Tensor
from data_classes import Point, Batch, Patch
import pandas as pd
ImageFile.LOAD_TRUNCATED_IMAGES = True

def crop_patches(im, rowcols, crop_size: int):
    '''
    Crop various sub-images (i.e. patches) from an image
    :param im: Pillow Image you need to crop into patches
    :param rowcols: list of Points representing the center of the sub-image to crop
    :param crop_size: size of the sub-image you need
    :return: patches : list of Pillow Images of crop_sizes
    '''
    patches = []
    try:
        # Define padding for points near the edges
        pad = crop_size
        im_padded = ImageOps.expand(im, pad, fill='white')

        # Adjust point coordinates to take padding into account
        adjusted_rowcols = [Point(p.x + pad, p.y + pad) for p in rowcols]
        for point in adjusted_rowcols:
            patch = crop_simple(im_padded, point, crop_size)
            patches.append(patch)

    except (OSError, IOError) as e:
        print(f"Error processing image: {e}")
        return []

    return patches


def crop_simple(im: Image, center: Point, crop_size: int) -> Image:
    '''
    Crop a single sub-image given image, center of the point and crop_size
    :param im: Pillow Image
    :param center: Point representing the center of the sub-image to crop
    :param crop_size: size of the sub-image you need
    :return: cropped image
    '''
    #print(center)
    upper = int(center.x - crop_size / 2)
    left = int(center.y - crop_size / 2)
    return im.crop((left, upper, left + crop_size, upper + crop_size))


def crop_images_in_patches(image_name: str, batch: Batch, crop_size: int, path_images_batch: str) -> tuple[
    Tensor, list[str | None]]:
    """

    :param image_name:
    :param batch:
    :param crop_size:
    :param path_images_batch:
    :return:
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    patch_tensor_list = []
    patch_labels_list = []
    # device="cpu"
    image_label = batch.get_image_labels(image_name)
    image_name, rowcols = image_label.get_points_per_image()
    #image_folder = os.path.join(output_folder, os.path.splitext(image_name)[0])
    #os.makedirs(image_folder, exist_ok=True)

    #patch_folder = 'test_patch'
    #os.makedirs(patch_folder, exist_ok=True)

    with Image.open(path_images_batch + '/' + image_name) as im:
        print("Opening image ", image_name)
        patches = crop_patches(im, rowcols, crop_size)
        print("Cropping patches ", len(patches))
        # write patches in a folder in order to debug
        for i, patch in enumerate(patches):
            #print("PATCH SIZE", patch.size)
            #(224,224)
            patch_name = f"patch_{i}.jpg"
            #patch_path = os.path.join(patch_folder, patch_name)
            #patch.save(patch_path)
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            patch_tensor = transform(patch).to(device)  # Normalize tensor
            #print("PATCH AFTER TRANSFORMATION", patch_tensor.shape)
            #(3,224,224)
            patch_tensor_list.append(patch_tensor)
            # Create Patch instance
            patch_point = rowcols[i]
            patch_label = image_label.get_label_at_point(patch_point)
            patch_labels_list.append(patch_label)
            new_patch = Patch(patch_point, patch_name, patch_label)

            image_label.add_patch(new_patch)

    patches_tensor = torch.stack(patch_tensor_list)
    #print("PATCHES TENSOR SHAPE: ", patches_tensor.shape)
    #(25,3,224,224)
    return patches_tensor, patch_labels_list


