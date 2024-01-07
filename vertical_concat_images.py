import os
import argparse
from PIL import Image


def concatenate_images_from_directory(directory_path, output_path):
    # Get a list of all files in the directory
    image_files = [
        f for f in os.listdir(directory_path)
        if f.lower().endswith('.png')]

    # Sort the list to ensure images are concatenated in the desired order
    image_files.sort()

    # Open each image
    img_list = [
        Image.open(os.path.join(directory_path, img))
        for img in image_files]

    # Get the width and height of the first image
    width, height = img_list[0].size

    # Create a new blank image with a height that can accommodate all input images vertically
    new_image = Image.new('RGB', (width, height * len(img_list)))

    # Paste each image vertically
    for i, img in enumerate(img_list):
        new_image.paste(img, (0, i * height))

    # Save the concatenated image
    new_image.save(output_path)


def main():
    # Create an argument parser
    parser = argparse.ArgumentParser(
        description='Concatenate PNG images in a directory vertically.')

    # Add the directory path argument
    parser.add_argument(
        'directory_path', type=str,
        help='Path to the directory containing PNG images')

    # Add the output path argument
    parser.add_argument(
        'output_path', type=str,
        help='Path for the vertically concatenated image output')

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the function to concatenate images vertically
    concatenate_images_from_directory(
        args.directory_path, args.output_path)


if __name__ == "__main__":
    main()
