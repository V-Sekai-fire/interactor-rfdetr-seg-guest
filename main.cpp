// rfdetr_seg.elf: RF-DETR instance segmentation (RFDETRSegNano) on ggml-rd.
//
// The graph is rf-detr-ggml's own (vendor/rf-detr-ggml, the one test_segmentation.cpp
// validates): DINOv2 backbone, projector, decoder with its in-graph top-k, segmentation
// head. What changes here is only where bytes come from. The guest has no filesystem
// (Gate 0F), so each GGUF's metadata is read through the pump (READ) and every weight
// is streamed by the host straight into the RenderingDevice buffer (UPLOAD): weights
// never enter the guest heap.
//
// Host side (project/gate_rfdetr_seg.gd, project/infer_host.gd):
//   rfdetr_attach(rd, total_mb)           once
//   rfdetr_start(model_dir, frames)       frames: space-separated host paths, each
//                                         312*312*3 float32 planar (C, H, W), bilinear
//                                         resized and ImageNet-normalised
//   rfdetr_pump(data)                     every frame until DONE / ERROR (rule 4)
//   rfdetr_result_size(), rfdetr_result_chunk(off, n)   the outputs, <= 8 MiB a call
// Per frame the result holds, back to back as float32:
//   boxes (100, 4) cx cy w h in [0, 1] | logits (100, 91), COCO ids, person = 1 |
//   masks (100, 78, 78) mask logits.
#include <api.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "backbone.h"
#include "decoder.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-rd.h"
#include "ggml.h"
#include "gguf.h"
#include "ops.h"
#include "projector.h"
#include "pump/pump.h"
#include "rd_compute.h"
#include "segmentation.h"

static rdc::Device g_dev;
static bool g_attached = false;
static std::string g_log;

static void logf_(const char *fmt, ...) __attribute__((format(printf, 1, 2)));
#include <cstdarg>
static void logf_(const char *fmt, ...) {
	char buf[1024];
	va_list ap;
	va_start(ap, fmt);
	std::vsnprintf(buf, sizeof buf, fmt, ap);
	va_end(ap);
	g_log += buf;
	g_log += '\n';
	std::printf("rfdetr_seg: %s\n", buf);
}

// --- ggml-rd hooks: the same four ggml_test uses -----------------------------------
static void hook_wait_gpu(void *) {
	pump::wait_gpu();
}

static void hook_coop(void *) {
	pump::coop();
}

static bool hook_upload(void *, const std::string &path, uint64_t file_offset, uint64_t bytes, ::RID rid,
		uint64_t dst_offset) {
	return pump::upload(path, file_offset, bytes, rid, dst_offset);
}

static bool hook_read(void *, const std::string &path, uint64_t file_offset, uint64_t bytes, void *dst) {
	const uint64_t chunk = uint64_t(16) << 20;
	for (uint64_t done = 0; done < bytes;) {
		const uint64_t n = std::min(bytes - done, chunk);
		std::vector<uint8_t> v = pump::read(path, file_offset + done, n);
		if (v.size() != n) {
			return false;
		}
		std::memcpy(static_cast<uint8_t *>(dst) + done, v.data(), size_t(n));
		done += n;
	}
	return true;
}

static void on_ggml_abort(const char *message) {
	std::fflush(stdout);
	pump::fail(std::string("ggml abort: ") + message);
}

// --- GGUF through the pump ------------------------------------------------------------
static size_t gguf_pump_read(void *userdata, void *output, uint64_t offset, size_t len) {
	const std::string &path = *static_cast<const std::string *>(userdata);
	std::vector<uint8_t> v = pump::read(path, offset, len);
	std::memcpy(output, v.data(), v.size());
	return v.size();
}

// Model::load_backend, with fopen/fread replaced by the pump: metadata by READ, each
// tensor by UPLOAD from its file offset into the RD buffer.
static bool load_gguf(Model &m, const std::string &path, ggml_backend_buffer_type_t buft) {
	ggml_context *c = nullptr;
	gguf_init_params gp = { /*no_alloc*/ true, /*ctx*/ &c };
	gguf_context *g = gguf_init_from_callback(gguf_pump_read, const_cast<std::string *>(&path), size_t(16) << 20,
			UINT64_MAX, gp);
	if (g == nullptr) {
		logf_("%s: gguf metadata did not parse", path.c_str());
		return false;
	}
	ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(c, buft);
	if (buf == nullptr) {
		logf_("%s: RD buffer allocation failed", path.c_str());
		gguf_free(g);
		return false;
	}
	m.bufs.push_back(buf);
	const size_t data_off = gguf_get_data_offset(g);
	size_t total = 0;
	int n = 0;
	for (ggml_tensor *t = ggml_get_first_tensor(c); t != nullptr; t = ggml_get_next_tensor(c, t)) {
		const int64_t idx = gguf_find_tensor(g, ggml_get_name(t));
		if (idx < 0) {
			logf_("%s: tensor %s missing from the index", path.c_str(), ggml_get_name(t));
			gguf_free(g);
			return false;
		}
		const size_t nbytes = ggml_nbytes(t);
		if (!ggml_backend_rd_tensor_upload(t, 0, path, data_off + gguf_get_tensor_offset(g, idx), nbytes)) {
			logf_("%s: upload of %s failed: %s", path.c_str(), ggml_get_name(t), ggml_backend_rd_last_error().c_str());
			gguf_free(g);
			return false;
		}
		m.weights[ggml_get_name(t)] = t;
		total += nbytes;
		n++;
	}
	m.ctx_w.push_back(c);
	gguf_free(g);
	logf_("%s: %d tensors, %.1f MB uploaded", path.c_str(), n, total / 1048576.0);
	return true;
}

// --- the model and its graph, built once ---------------------------------------------
struct Seg {
	Model m;
	ggml_backend_t be = nullptr;
	ggml_gallocr_t alloc = nullptr;
	ggml_cgraph *gf = nullptr;
	ggml_tensor *x = nullptr;
	ggml_tensor *proposals = nullptr;
	ggml_tensor *boxes = nullptr;
	ggml_tensor *logits = nullptr;
	ggml_tensor *mask = nullptr;
	std::vector<float> prop;
	bool ready = false;
};
static Seg *g_seg = nullptr;

static const int64_t RES = 312;
static const size_t N_PX = size_t(RES) * RES * 3;

static bool build(const std::string &dir) {
	g_seg = new Seg();
	Seg &s = *g_seg;
	ggml_backend_reg_t reg = ggml_backend_rd_reg();
	if (ggml_backend_reg_dev_count(reg) == 0) {
		logf_("no ggml-rd device (attach a RenderingDevice first)");
		return false;
	}
	ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
	s.be = ggml_backend_dev_init(dev, nullptr);
	ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
	for (const char *part : { "backbone", "projector", "decoder", "segmentation" }) {
		if (!load_gguf(s.m, dir + "/rf-detr-seg-nano-" + part + ".gguf", buft)) {
			return false;
		}
	}

	BackboneParams bp;
	bp.hidden = 384; bp.n_layer = 12; bp.n_head = 6; bp.patch_size = 12; bp.n_register = 0; bp.num_windows = 1;
	bp.window_block_indexes = { 0, 1, 2, 4, 5, 7, 8, 10, 11 };
	bp.out_feature_indexes = { 2, 5, 8, 11 };
	DecoderParams dp;
	dp.hidden_dim = 256; dp.dec_layers = 4; dp.sa_nheads = 8; dp.ca_nheads = 16; dp.dec_n_points = 2;
	dp.num_queries = 100; dp.num_classes = 91; dp.gw = 26; dp.gh = 26;
	SegmentationParams sp;
	sp.hidden_dim = 256; sp.num_blocks = 4; sp.downsample_ratio = 4; sp.image_w = 312; sp.image_h = 312;

	const size_t max_nodes = 200000;
	ggml_init_params ip = { ggml_tensor_overhead() * max_nodes + ggml_graph_overhead_custom(max_nodes, false), nullptr, true };
	s.m.ctx_g = ggml_init(ip);
	s.x = ggml_new_tensor_4d(s.m.ctx_g, GGML_TYPE_F32, RES, RES, 3, 1);
	ggml_set_name(s.x, "pixel_values");
	ggml_set_input(s.x);
	std::vector<ggml_tensor *> taps = dinov2_backbone(s.m, s.x, bp);
	ggml_tensor *fused = projector_p4(s.m, taps, 256);
	ggml_tensor *memory = ggml_reshape_3d(s.m.ctx_g, fused, dp.gw * dp.gh, dp.hidden_dim, 1);
	memory = ggml_cont(s.m.ctx_g, ggml_permute(s.m.ctx_g, memory, 1, 0, 2, 3));
	DecoderOutput dout = rfdetr_decoder(s.m, memory, dp); // in-graph top-k
	s.mask = segmentation_head(s.m, fused, dout.hidden_states, sp).back();
	s.boxes = dout.pred_boxes;
	s.logits = dout.pred_logits;
	s.proposals = dout.output_proposals;
	if (!s.m.ok()) {
		logf_("missing weights, first: %s", s.m.missing_tensors().front().c_str());
		return false;
	}
	s.gf = ggml_new_graph_custom(s.m.ctx_g, max_nodes, false);
	for (ggml_tensor *t : { s.mask, s.boxes, s.logits }) {
		ggml_build_forward_expand(s.gf, t);
	}
	s.alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(s.be));
	if (!ggml_gallocr_alloc_graph(s.alloc, s.gf)) {
		logf_("graph allocation failed");
		return false;
	}
	s.prop = output_proposals_data(dp.gw, dp.gh);
	logf_("graph: %d nodes", ggml_graph_n_nodes(s.gf));
	s.ready = true;
	return true;
}

// --- the job ---------------------------------------------------------------------------
struct Job {
	std::string dir;
	std::vector<std::string> frames;
};
static Job g_job;
static std::vector<uint8_t> g_result;

static void seg_job(void *) {
	if (g_seg == nullptr || !g_seg->ready) {
		if (!build(g_job.dir)) {
			pump::fail("rfdetr_seg: model build failed:\n" + g_log);
		}
	}
	Seg &s = *g_seg;
	for (const std::string &f : g_job.frames) {
		ggml_backend_rd_ensure_idle();
		if (!ggml_backend_rd_tensor_upload(s.x, 0, f, 0, N_PX * sizeof(float))) {
			logf_("%s: frame upload failed: %s", f.c_str(), ggml_backend_rd_last_error().c_str());
			continue;
		}
		ggml_backend_tensor_set(s.proposals, s.prop.data(), 0, s.prop.size() * sizeof(float));
		if (ggml_backend_graph_compute(s.be, s.gf) != GGML_STATUS_SUCCESS) {
			logf_("%s: compute failed: %s", f.c_str(), ggml_backend_rd_last_error().c_str());
			continue;
		}
		for (ggml_tensor *t : { s.boxes, s.logits, s.mask }) {
			const size_t nb = ggml_nbytes(t);
			const size_t at = g_result.size();
			g_result.resize(at + nb);
			ggml_backend_tensor_get(t, g_result.data() + at, 0, nb); // waits on a later frame (rule 4)
		}
		logf_("%s: done", f.c_str());
	}
}

static std::vector<std::string> split_ws(const std::string &s) {
	std::vector<std::string> r;
	size_t i = 0;
	while (i < s.size()) {
		while (i < s.size() && s[i] == ' ') {
			++i;
		}
		size_t j = i;
		while (j < s.size() && s[j] != ' ') {
			++j;
		}
		if (j > i) {
			r.push_back(s.substr(i, j - i));
		}
		i = j;
	}
	return r;
}

// --- API -------------------------------------------------------------------------------
static Variant rfdetr_attach(Object rd, int64_t total_mb) {
	if (rd.is_valid()) {
		g_dev.adopt(rd);
	}
	g_attached = g_dev.ok();
	static bool registered = false;
	if (!registered) {
		registered = true;
		ggml_backend_register(ggml_backend_rd_reg());
	}
	ggml_backend_rd_attach(g_attached ? &g_dev : nullptr, size_t(total_mb) << 20);
	ggml_rd_hooks h;
	h.wait_gpu = hook_wait_gpu;
	h.coop = hook_coop;
	h.upload = hook_upload;
	h.read = hook_read;
	ggml_backend_rd_set_hooks(h);
	ggml_set_abort_callback(on_ggml_abort);
	return Variant(String(g_attached ? "attached device=" + g_dev.device_name() : std::string("no RD device")));
}

// "K=V K2=V2": set for this job; the GGML_RD_* switches not named are unset.
static void apply_env(const std::string &env) {
	for (const char *k : { "GGML_RD_FAULT", "GGML_RD_BARRIER_ALL", "GGML_RD_TIMESTAMPS" }) {
		unsetenv(k);
	}
	for (const std::string &kv : split_ws(env)) {
		const size_t eq = kv.find('=');
		if (eq != std::string::npos) {
			setenv(kv.substr(0, eq).c_str(), kv.substr(eq + 1).c_str(), 1);
		}
	}
}

static Variant rfdetr_start(String model_dir, String frames, String env) {
	apply_env(env.utf8());
	g_log.clear();
	g_result.clear();
	g_job.dir = model_dir.utf8();
	g_job.frames = split_ws(frames.utf8());
	if (!pump::start(&seg_job, nullptr, size_t(8) << 20)) {
		return Variant(String("FAIL a job is running"));
	}
	return Variant(String("STARTED " + std::to_string(g_job.frames.size()) + " frames"));
}

static Variant rfdetr_pump(PackedArray<uint8_t> in) {
	return pump::step(in);
}

static Variant rfdetr_result_size() {
	return Variant(int64_t(g_result.size()));
}

static Variant rfdetr_result_chunk(int64_t offset, int64_t bytes) {
	const size_t off = size_t(std::max<int64_t>(offset, 0));
	if (off >= g_result.size()) {
		return Variant(PackedArray<uint8_t>(std::vector<uint8_t>()));
	}
	const size_t n = std::min<size_t>({ size_t(std::max<int64_t>(bytes, 0)), size_t(8) << 20, g_result.size() - off });
	return Variant(PackedArray<uint8_t>(g_result.data() + off, n));
}

static Variant rfdetr_output() {
	return Variant(String(g_log));
}

static Variant rfdetr_stats() {
	std::string r = ggml_backend_rd_stats();
	r += " rule4_same_frame_syncs=" + std::to_string(g_dev.same_frame_syncs());
	r += " syncs=" + std::to_string(g_dev.syncs());
	r += " submits=" + std::to_string(g_dev.submits());
	return Variant(String(r));
}

int main() {
	ADD_API_FUNCTION(rfdetr_attach, "String", "Object rd, int total_mb", "Attach the host's RenderingDevice");
	ADD_API_FUNCTION(rfdetr_start, "String", "String model_dir, String frames, String env", "Segment preprocessed frames on the pump (env: K=V GGML_RD_* switches)");
	ADD_API_FUNCTION(rfdetr_pump, "Array", "PackedByteArray data", "Resume the job once: [header, text, rid]");
	ADD_API_FUNCTION(rfdetr_result_size, "int", "", "Bytes of results so far");
	ADD_API_FUNCTION(rfdetr_result_chunk, "PackedByteArray", "int offset, int bytes", "Result bytes (<= 8 MiB)");
	ADD_API_FUNCTION(rfdetr_output, "String", "", "What the job logged");
	ADD_API_FUNCTION(rfdetr_stats, "String", "", "ggml-rd and rd_compute counters");
	halt();
}
