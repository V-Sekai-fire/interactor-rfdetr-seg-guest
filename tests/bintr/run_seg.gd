# rfdetr_seg.elf in a Sandbox: probe loads it, run segments one frame; tools/check_bintr.exs reads the "key value" lines.
#   godot --path tests/bintr --script run_seg.gd -- --mode=probe|run --translate=yes|no --elf=E --bintr=D
#         [--timeout=U --models=M --frame=F --out=O]
extends SceneTree

const WAIT_GPU := 1
const READ := 2
const COOP := 4
const DONE := 5
const ERROR := 6


func _initialize() -> void:
	var a := {}
	for arg in OS.get_cmdline_user_args():
		var kv := arg.trim_prefix("--").split("=", true, 1)
		a[kv[0]] = kv[1] if kv.size() > 1 else ""
	quit(_run(a))


func _run(a: Dictionary) -> int:
	if not ClassDB.class_exists("Sandbox"):
		return _fail("the Sandbox class is not registered")
	ProjectSettings.set_setting("sandbox/binary_translation/cache_dir", a.bintr + "/")
	ProjectSettings.set_setting("sandbox/binary_translation/enabled", a.get("translate") == "yes")
	var sb = ClassDB.instantiate("Sandbox")
	sb.memory_max = 1024
	sb.references_max = 4096
	sb.allocations_max = 1000000
	sb.execution_timeout = int(a.get("timeout", "214577"))
	var elf := FileAccess.get_file_as_bytes(a.elf)
	if elf.is_empty():
		return _fail("%s did not read" % a.elf)
	sb.load_buffer(elf)
	if not sb.has_function("rfdetr_start"):
		return _fail("%s did not load (no rfdetr_start)" % a.elf.get_file())
	print("hash %08X" % (int(sb.get_translation_hash()) & 0xFFFFFFFF))
	print("translated %s" % ("yes" if sb.is_binary_translated() else "no"))
	if a.get("mode") != "run":
		sb.free()
		return 0

	print("attach %s" % str(sb.vmcall("rfdetr_attach", 0, 1024)).replace(" ", "_"))
	print("start %s" % str(sb.vmcall("rfdetr_start", a.models, a.frame, "")).replace(" ", "_"))
	var feed := PackedByteArray()
	var pumps := 0
	var reads := 0
	var vm_us := 0
	while true:
		var t0 := Time.get_ticks_usec()
		var r = sb.vmcall("rfdetr_pump", feed)
		vm_us += Time.get_ticks_usec() - t0
		pumps += 1
		feed = PackedByteArray()
		if typeof(r) != TYPE_ARRAY or r.size() < 3:
			return _fail("rfdetr_pump returned %s" % str(r))
		var hdr: PackedInt64Array = r[0]
		if hdr[0] == DONE:
			break
		if hdr[0] == ERROR:
			return _fail("guest: %s" % str(r[1]).replace("\n", " | "))
		if hdr[0] == READ:
			var f := FileAccess.open(str(r[1]), FileAccess.READ)
			if f == null:
				return _fail("READ %s: %s" % [r[1], error_string(FileAccess.get_open_error())])
			f.seek(hdr[1])
			feed = f.get_buffer(hdr[2] if hdr[2] > 0 else f.get_length() - hdr[1])
			reads += 1
		elif hdr[0] != COOP and hdr[0] != WAIT_GPU:
			return _fail("unknown request kind %d" % hdr[0])
	print("pumps %d" % pumps)
	print("reads %d" % reads)
	print("vm_ms %d" % (vm_us / 1000))
	print("guest_log %s" % str(sb.vmcall("rfdetr_output")).strip_edges().replace("\n", " | "))
	var size := int(sb.vmcall("rfdetr_result_size"))
	var out := PackedByteArray()
	while out.size() < size:
		var chunk: PackedByteArray = sb.vmcall("rfdetr_result_chunk", out.size(), size - out.size())
		if chunk.is_empty():
			return _fail("rfdetr_result_chunk returned nothing at %d of %d" % [out.size(), size])
		out.append_array(chunk)
	var w := FileAccess.open(a.out, FileAccess.WRITE)
	if w == null:
		return _fail("cannot write %s" % a.out)
	w.store_buffer(out)
	w.close()
	print("result_bytes %d" % out.size())
	print("translated_after %s" % ("yes" if sb.is_binary_translated() else "no"))
	sb.free()
	return 0


func _fail(why: String) -> int:
	print("FAIL %s" % why)
	return 1
