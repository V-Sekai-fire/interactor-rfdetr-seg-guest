# interactor-rfdetr-seg-guest

RF-DETR instance segmentation on ggml-rd as a godot-sandbox guest.

Split out of `interactor-dress-on` at `310b52e` with its history (`git subtree`). It sits at `3-interactor/rfdetr-seg-guest` in the goal manifest (`contract-manifest-taskweft`), and finds the repositories it builds against as sibling checkouts at their manifest paths. `transport-meshing-pen` builds the guest ELFs (`build.sh`, `tools/build.exs`).
