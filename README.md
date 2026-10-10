# interactor-rfdetr-seg-guest

RF-DETR instance segmentation on ggml-rd, built as a sandboxed guest program that the engine host feeds frames and weights.

## What it is for

The guest has no filesystem, so the host streams each model's weights straight into a GPU buffer and pumps frames through it, and the guest returns boxes, class logits and mask logits per frame. RFD 2272 owns the design.

## Build and run

`elixir tools/build.exs` builds `rfdetr_seg.elf` with `contract-guest-runtime`'s shared guest build, against the sibling checkouts the goal manifest places beside this one; the RF-DETR graph is vendored from `interactor-rf-detr-ggml` under `vendor/rf-detr-ggml`. `elixir tools/check_bintr.exs --elf=<rfdetr_seg.elf>` runs it under godot-sandbox with binary translation and without, and compares the two; CI compares its translated run with the interpreted run recorded in `tests/bintr/interpreted`.

## Licence

MIT. See [LICENSE](LICENSE).
