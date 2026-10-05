# rfdetr_seg.elf: RF-DETR instance segmentation (RFDETRSegNano) on ggml-rd, the graph from vendor/rf-detr-ggml unchanged.
get_filename_component(RFDETR_SEG_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(_rf ${RFDETR_SEG_ROOT}/vendor/rf-detr-ggml/src)

add_stage_elf(rfdetr_seg
	${RFDETR_SEG_ROOT}/guest/rfdetr_seg/main.cpp
	${_rf}/ops.cpp
	${_rf}/backbone.cpp
	${_rf}/projector.cpp
	${_rf}/deform_attn.cpp
	${_rf}/decoder.cpp
	${_rf}/keypoints.cpp
	${_rf}/segmentation.cpp
)
target_include_directories(rfdetr_seg PRIVATE ${GUEST_RUNTIME_ROOT}/guest ${_rf})
target_link_libraries(rfdetr_seg PRIVATE ggml_rd pump ggml ggml-base)
