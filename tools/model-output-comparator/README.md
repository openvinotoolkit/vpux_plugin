# comparator of different inference output

### Summary

The tool takes 2 inference outputs and shows difference between them.
Have a nice debug :)

### Usage


The tool has the following required command line arguments:

1. path to reference output
2. path to actual output
3. path to W, H, C data file, contained something like "64 64 512", or W H C values instead

Tool tries to choose the best layout automatically, but if difference is big
sometimes heuristics doesn't work, so you can try to force some layout by uncomment appropriate line:
```
#force some layout, if needed
#kmb_img = kmb_img1
#kmb_img = kmb_img2
#kmb_img = kmb_img3
#kmb_img = kmb_img4
```

You can change number of channels to visualize using max_columns variable


### Example

python3 bin_diff.py cpu_output kmb_output WHC.txt

### The output

Several channels difference in format 2x2 images per output channel.

2 top images - raw reference and actual, 2 bottom - diff in BWR color map and error distribution.
