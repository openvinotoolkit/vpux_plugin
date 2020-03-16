#ifndef _PLG_OT_H_
#define _PLG_OT_H_
//#include <ImgFrame.h>
#include <stdio.h>
#include <vector>
#include "Flic.h"
#include "Message.h"
#include "Pool.h"
#include "PlgOTTypes.h"

class PlgOT : public PluginStub{
  public:
    PlgOT() : PluginStub("PlgOT") {}

    SReceiver<ImgFramePtr> in0;
    SReceiver<vpuot::DetectedObjects> in1;
    SReceiver<float> in2;
    MReceiver<vpuot::OutObjectsPtr> in3;

    MSender<vpuot::OutObjectsPtr> out;
    int32_t  Create(vpuot::TrackType ot_type, int32_t max_objects, float mask_padding_thickness);
};
#endif
