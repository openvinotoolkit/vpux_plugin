#
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#
import sys
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('dark_background')

kmb_filename = "InferenceManagerDemo/output-0.bin"
cpu_filename = "InferenceManagerDemo/cpu-output-0.bin"
if len(sys.argv) >= 3:
    cpu_filename = sys.argv[1]
    kmb_filename = sys.argv[2]

if len(sys.argv) == 6:
    W = int(sys.argv[3])
    H = int(sys.argv[4])
    C = int(sys.argv[5])
else:
    WHC_file = "InferenceManagerDemo/WHC.txt"
    if len(sys.argv) == 4:
        WHC_file = sys.argv[3]
    with open(WHC_file, "r") as myfile:
        data = myfile.read()
        W = int(data.split(" ")[0])
        H = int(data.split(" ")[1])
        C = int(data.split(" ")[2])

cpu_img = np.fromfile(cpu_filename, np.float32)
kmb_img = np.fromfile(kmb_filename, np.float32)
if 2 * len(kmb_img) == len(cpu_img):
    kmb_img = np.fromfile(kmb_filename, np.float16)
    kmb_img = kmb_img.astype(np.float32)
    print("KMB input treated as float16")


print("KMB: min {min} max {max}".format(min=np.min(kmb_img), max=np.max(kmb_img)))
print("CPU: min {min} max {max}".format(min=np.min(cpu_img), max=np.max(cpu_img)))

plt.figure(figsize=(16, 9))
columns = 1
rows = 1
max_columns = 4
for columns in range(1, max_columns + 1):
    rows = int(round((columns * 9.0) / 16.0))
    if columns * rows >= C:
        break

cpu_img = cpu_img.reshape((W, H, C), order='F')  # C F A
cpu_img = np.fliplr(np.rot90(np.rot90(np.rot90(cpu_img))))

kmb_img1 = kmb_img.reshape((W, H, C), order='F')  # C F A
kmb_img1 = np.fliplr(np.rot90(np.rot90(np.rot90(kmb_img1))))

kmb_img2 = kmb_img.reshape((W, H, C), order='C')  # C F A
kmb_img2 = np.fliplr(np.rot90(np.rot90(np.rot90(kmb_img2))))

kmb_img3 = kmb_img.reshape((H, W, C), order='F')  # C F A
kmb_img4 = kmb_img.reshape((H, W, C), order='C')  # C F A

d1 = np.sum(np.abs(cpu_img - kmb_img1))
d2 = np.sum(np.abs(cpu_img - kmb_img2))
d3 = np.sum(np.abs(cpu_img - kmb_img3))
d4 = np.sum(np.abs(cpu_img - kmb_img4))

print("Diff1 cpu kmb: min {min} max {max} avg {avg} sq ave {sa}".format(min=np.min(cpu_img - kmb_img1), max=np.max(
    cpu_img - kmb_img1), avg=np.average(cpu_img - kmb_img1), sa=(np.square(cpu_img - kmb_img1)).mean(axis=None)))
print("Diff2 cpu kmb: min {min} max {max} avg {avg} sq ave {sa}".format(min=np.min(cpu_img - kmb_img2), max=np.max(
    cpu_img - kmb_img2), avg=np.average(cpu_img - kmb_img2), sa=(np.square(cpu_img - kmb_img2)).mean(axis=None)))
print("Diff3 cpu kmb: min {min} max {max} avg {avg} sq ave {sa}".format(min=np.min(cpu_img - kmb_img3), max=np.max(
    cpu_img - kmb_img3), avg=np.average(cpu_img - kmb_img3), sa=(np.square(cpu_img - kmb_img3)).mean(axis=None)))
print("Diff4 cpu kmb: min {min} max {max} avg {avg} sq ave {sa}".format(min=np.min(cpu_img - kmb_img4), max=np.max(
    cpu_img - kmb_img4), avg=np.average(cpu_img - kmb_img4), sa=(np.square(cpu_img - kmb_img4)).mean(axis=None)))

if d1 <= d2 and d1 <= d3 and d1 <= d4:
    kmb_img = kmb_img1
    print("Diff1 choosed")
if d2 <= d1 and d2 <= d3 and d2 <= d4:
    kmb_img = kmb_img2
    print("Diff2 choosed")
if d3 <= d1 and d3 <= d2 and d3 <= d4:
    kmb_img = kmb_img3
    print("Diff3 choosed")
if d4 <= d1 and d4 <= d2 and d4 <= d3:
    kmb_img = kmb_img4
    print("Diff4 choosed")

# force some layout, if needed
# kmb_img = kmb_img1
# kmb_img = kmb_img2
# kmb_img = kmb_img3
# kmb_img = kmb_img4

diff_img = cpu_img - kmb_img
print("Diff cpu kmb: min {min} max {max} avg {avg} sq ave {sa}".format(min=np.min(diff_img),
      max=np.max(diff_img), avg=np.average(diff_img), sa=(np.square(diff_img)).mean(axis=None)))


def draw_cpu_fm(cpu_fmps, kmb_fmps, channel):
    return cpu_fmps[:, :, channel]


def draw_kmb_fm(cpu_fmps, kmb_fmps, channel):
    return kmb_fmps[:, :, channel]


for row in range(rows):
    for col in range(columns):
        c = row * columns + col
        if c >= C:
            continue
        if rows * columns < C and row >= rows // 2:
            c = C - 1 - (columns - col - 1) - (rows - row - 1) * columns

        for sub_r in range(2):
            for sub_c in range(2):
                i = row * 2 * columns * 2 + col * 2 + sub_r * columns * 2 + sub_c
                plt.subplot2grid((rows * 45, columns * 45), (row * 45 + sub_r * 21 +
                                 2, col * 45 + sub_c * 21 + 2), rowspan=20, colspan=20)

                if sub_r == 0:
                    if sub_c == 0:
                        img = draw_cpu_fm(cpu_img, kmb_img, c)
                        cmap = 'gray'
                        max_val = max(np.max(cpu_img[:, :, c]), np.max(kmb_img[:, :, c]))
                        min_val = min(np.min(cpu_img[:, :, c]), np.min(kmb_img[:, :, c]), 0)
                        imshow = plt.imshow(img, cmap=cmap, vmin=min_val, vmax=max_val)
                    else:
                        img = draw_kmb_fm(cpu_img, kmb_img, c)
                        cmap = 'gray'
                        max_val = max(np.max(cpu_img[:, :, c]), np.max(kmb_img[:, :, c]))
                        min_val = min(np.min(cpu_img[:, :, c]), np.min(kmb_img[:, :, c]), 0)
                        imshow = plt.imshow(img, cmap=cmap, vmin=min_val, vmax=max_val)
                else:
                    if sub_c == 0:
                        img = diff_img[:, :, c]
                        cmap = 'bwr'
                        max_val = max(np.max(cpu_img), np.max(kmb_img))
                        min_val = min(np.min(cpu_img), np.min(kmb_img), 0)
                        max_val = (max_val - min_val) / (25.6) / 1
                        min_val = -max_val
                        imshow = plt.imshow(img, cmap=cmap, vmin=min_val, vmax=max_val)
                    else:
                        map_size = 128
                        half_map_size = map_size // 2
                        img = np.zeros((map_size * map_size))
                        max_val = np.max(cpu_img[:, :, c])
                        min_val = np.min(cpu_img[:, :, c])
                        max_diff = max(np.max(diff_img[:, :, c]), -np.min(diff_img[:, :, c]))
                        min_val = min(min_val, 0)
                        line_len = map_size // 20
                        if (min_val == 0 and max_val == 0) or max_diff == 0:
                            img = np.reshape(img, (map_size, map_size))
                            cmap = 'gray'
                            plt.imshow(img, cmap=cmap, interpolation='bicubic')  # , vmin=0, vmax=W)
                            plt.axis('off')
                            continue

                        x_coord = ((((diff_img[:, :, c]) * (half_map_size - line_len)) /
                                   (max_diff)) + half_map_size - line_len // 2).astype(int)
                        y_coord = ((((cpu_img[:, :, c] - min_val) * (map_size - 1)) / (max_val - min_val))).astype(int)
                        x_coord = np.reshape(x_coord, (W * H))
                        y_coord = np.reshape(y_coord, (W * H))

                        for l in range(line_len):
                            img += np.bincount(y_coord * map_size + x_coord + l, minlength=map_size * map_size)
                        img = np.power(img, 0.25)
                        img = np.reshape(img, (map_size, map_size))

                        cmap = 'gray'
                        plt.imshow(img, cmap=cmap, interpolation='bicubic')  # , vmin=0, vmax=W)

                plt.axis('off')

plt.subplots_adjust(0, 0, 1, 1, 0, 0)
plt.show()
