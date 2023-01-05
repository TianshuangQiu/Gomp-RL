import imageio
import glob
import os
import pdb

for i in range(9):

    writer = imageio.get_writer(f"~/gomp/videos/env{i}.mp4", fps=30)
    file_names = [
        file
        for file in glob.glob(os.path.join("graphics_images", f"rgb_env{i}_cam1*.png"))
    ]
    file_names.sort()
    # pdb.set_trace()
    for file in file_names:
        im = imageio.imread(file)
        writer.append_data(im)
    writer.close()
