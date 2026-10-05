# interactor-rfdetr-seg-guest

RF-DETR instance segmentation on ggml-rd, built as a sandboxed guest program that the engine host feeds frames and weights.

## What it is for

The guest has no filesystem, so the host streams each model's weights straight into a GPU buffer and pumps frames through it, and the guest returns boxes, class logits and mask logits per frame. RFD 2272 owns the design.

## Build and run

The guest has no build of its own here, and no other repository builds it now. The source is `guest/rfdetr_seg/main.cpp`, and a build has to supply ggml-rd and the frame-pump headers it includes.

## Licence

The licence is not stated.
