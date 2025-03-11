# Examples:

This is the main function signature: `main(image_path: str, block_size=20, color=True, fast=True, height_reduction_factor=0.6)`

## Here are some examples:

`main('image.jpg')`

`main('image.jpg', block_size=8)`

`main('image.jpg', color=False)`


# Comments
- Fast version requires scipy. Howerver it won't raise an error if you try to run `main` with `fast=True`. You will just get a warning and it will run as if `fast` was set to `False`.
- It is recommended not to change too much (0.4 - 0.6) the hight_reduction_factor parameter, otherwise the result will be distorted.
