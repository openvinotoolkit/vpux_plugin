# edit xml tool

## Summary

The tool cuts the network to the specified layer and creates new IR.

## Prerequsites

The tool works with IR v10.

It works with python library lxml. The library needs installing the following way:
pip3 install lxml

## Usage


The tool has the following required command line arguments:

* `-m <path to the IR file with name and extension>` - path to IR file with name and extension:
* `-l <name of the layer>` - name of the layer which is supposed to be the output


## Example

python3 edit_xml.py -m `name-of-original-file`.xml -l `name-of-the-layer`

### The output

The output file is expected to b the IR based on the original and ended by the specified layer.
`name-of-original-file`-cut-`name-of-the output-layer`.xml
