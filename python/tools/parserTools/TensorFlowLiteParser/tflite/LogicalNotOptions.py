# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers

class LogicalNotOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsLogicalNotOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = LogicalNotOptions()
        x.Init(buf, n + offset)
        return x

    # LogicalNotOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

def LogicalNotOptionsStart(builder): builder.StartObject(0)
def LogicalNotOptionsEnd(builder): return builder.EndObject()
