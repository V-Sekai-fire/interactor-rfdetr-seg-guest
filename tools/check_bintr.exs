# rfdetr_seg.elf under godot-sandbox's libriscv on one fixed frame, natively translated and interpreted; every check has a
# control that must fail. CI runs the two parts on two runners and compares their directories in a third job.
#   elixir tools/check_bintr.exs --elf=<rfdetr_seg.elf> [--part=all|translated|interpreted] [--work=<dir>]
#   elixir tools/check_bintr.exs --compare=<translated dir>,<interpreted dir>
defmodule CheckBintr do
  @root Path.expand("..", __DIR__)
  @project Path.join(@root, "tests/bintr")
  @godot_build "https://github.com/V-Sekai-fire/service-godot-build/releases/download"
  @models_url "https://github.com/V-Sekai-fire/interactor-rf-detr-ggml/releases/download/v0.1.0-dev"
  @engine %{
    linux: {"v20260930-double.1", "godot.linuxbsd.editor.double.x86_64", "37c6962fcc76d1773a596440f310f52ad5cf18011e74a712a5b465c3240e919b"},
    macos: {"v20260930-double.1", "godot.macos.editor.double.arm64", "5f6f2df33708d8f522e0cb5f40af58bd44c1eccb2dda7128081f4056b1de8dce"}
  }
  @addon %{
    linux: {"v20261002-addon.1", "libgodot_riscv.linux.template_release.double.x86_64.so", "1cce3f2d207d4c7d625ed6ed6610fd7bba4a0c6301b242789ed2eecde5ce8f62"},
    macos: {"v20261002-addon.1", "libgodot_riscv.macos.template_release.double.universal", "5aa0fe75cc80e98473a4a6dfa57ada3f3f6d61d76a2c3f01073f36d0e36358d0"}
  }
  @models [
    {"rf-detr-seg-nano-backbone.gguf", "bec28e8f6e1ab925661370b3f3c881ea6e19e6d07f43cf52be45d1a88f186cb8"},
    {"rf-detr-seg-nano-projector.gguf", "82b0d7a64700aa231ea7987d5a636c4c6b005f401408d0edaf4538293a1c3f8c"},
    {"rf-detr-seg-nano-decoder.gguf", "621a4dbd88cde51cc9c997a34ba1f40888e82f78cbaa2ac88ebf618c58a3bd43"},
    {"rf-detr-seg-nano-segmentation.gguf", "443f51f956feb63f708589fbe5f478da4717a23e80d855ef0bdeeb7421dbe040"}
  ]
  # Boxes (100, 4), logits (100, 91) and masks (100, 78, 78), float32, back to back.
  @result_floats 100 * 4 + 100 * 91 + 100 * 78 * 78
  # sha256 of the outputs on the fixed frame, measured on macOS arm64 with the pinned engine and addon.
  @expected "94b4dae9c133a0fbcc65f6639e305c078d76f92297841b2f48cccbcb926ff0c1"
  @min_speedup 3.0
  @wall_s %{probe: 120, translated: 420, interpreted: 540}

  def main(argv) do
    {kv, _, _} = OptionParser.parse(argv, strict: [elf: :string, part: :string, work: :string, compare: :string])
    results =
      case kv[:compare] do
        nil -> parts(kv)
        dirs -> apply(&compare/2, Enum.map(String.split(dirs, ","), &Path.expand/1))
      end
    failed = Enum.count(results, &(&1 != :ok))
    say("#{length(results) - failed} of #{length(results)} checks pass")
    if failed > 0 or results == [], do: System.halt(1)
  end

  defp parts(kv) do
    elf = Path.expand(kv[:elf] || fail("no --elf"))
    File.exists?(elf) || fail("no #{elf}")
    work = Path.expand(kv[:work] || Path.join(@root, "build/bintr"))
    host = host()
    env = setup(host, work)
    t = Path.join(work, "translated")
    i = Path.join(work, "interpreted")
    case kv[:part] || "all" do
      "translated" -> translated(env, elf, t)
      "interpreted" -> interpreted(env, elf, i)
      "all" -> translated(env, elf, t) ++ interpreted(env, elf, i) ++ compare(t, i)
      other -> fail("unknown --part=#{other}")
    end
  end

  # --- the two runs ----------------------------------------------------------------

  defp translated(env, elf, dir) do
    reset(dir)
    lib = Path.join(dir, "bintr")
    src = Path.join(dir, "bintr-c")
    File.mkdir_p!(lib)
    File.mkdir_p!(src)
    emit = godot(env, :probe, ["--mode=probe", "--translate=yes", "--elf=#{elf}", "--bintr=#{lib}"], [{"GODOT_SANDBOX_BINTR_EMIT", src}])
    hash = emit["hash"] || fail("the emit probe printed no hash:\n#{emit.out}")
    c = Path.join(src, "bintr-#{hash}.c")
    File.exists?(c) || fail("the addon wrote no #{Path.basename(c)} (#{inspect(File.ls!(src))})")
    say("translation: #{Path.basename(c)}, #{File.stat!(c).size} bytes of C")
    compile(env, c, Path.join(lib, "bintr-#{hash}#{env.suffix}"))

    loaded = godot(env, :probe, ["--mode=probe", "--translate=yes", "--elf=#{elf}", "--bintr=#{lib}"])
    disabled = godot(env, :probe, ["--mode=probe", "--translate=no", "--elf=#{elf}", "--bintr=#{lib}"])
    absent = Path.join(dir, "no-bintr")
    File.mkdir_p!(absent)
    missing = godot(env, :probe, ["--mode=probe", "--translate=yes", "--elf=#{elf}", "--bintr=#{absent}"])
    flipped = Path.join(dir, "flipped.elf")
    File.write!(flipped, flip_text(File.read!(elf)))
    flip = godot(env, :probe, ["--mode=probe", "--translate=yes", "--elf=#{flipped}", "--bintr=#{lib}"])
    truncated = Path.join(dir, "truncated.elf")
    bytes = File.read!(elf)
    File.write!(truncated, binary_part(bytes, 0, div(byte_size(bytes), 2)))
    trunc = godot(env, :probe, ["--mode=probe", "--translate=yes", "--elf=#{truncated}", "--bintr=#{lib}"])

    run = full_run(env, elf, dir, "yes", lib)
    [
      check("the loaded program carries the native translation (hash #{hash})", loads_translated(loaded, hash)),
      check("control: with binary translation disabled it is refused", refused(loads_translated(disabled, hash))),
      check("control: with no library for the hash it is refused", refused(loads_translated(missing, hash))),
      check("control: an ELF with one .text byte flipped is refused", refused(loads_translated(flip, hash))),
      check("control: an ELF cut in half is refused", refused(loads_translated(trunc, hash))),
      check("the translated run finished on its translation", ran(run, hash, "yes"))
    ]
  end

  defp interpreted(env, elf, dir) do
    reset(dir)
    lib = Path.join(dir, "bintr")
    File.mkdir_p!(lib)
    run = full_run(env, elf, dir, "no", lib)
    [check("the interpreted run finished interpreted", ran(run, run["hash"], "no"))]
  end

  defp full_run(env, elf, dir, translate, lib) do
    out = Path.join(dir, "out.f32")
    kind = if translate == "yes", do: :translated, else: :interpreted
    r = godot(env, kind, ["--mode=run", "--translate=#{translate}", "--elf=#{elf}", "--bintr=#{lib}",
                          "--models=#{env.models}", "--frame=#{env.frame}", "--out=#{out}"])
    report = Map.drop(r, [:out]) |> Map.put("elf_sha256", sha256(File.read!(elf))) |> Map.put("rc", to_string(r.rc))
    File.write!(Path.join(dir, "report.txt"), Enum.map_join(Enum.sort(report), "", fn {k, v} -> "#{k} #{v}\n" end))
    File.write!(Path.join(dir, "godot.log"), r.out)
    say("#{kind}: rc #{r.rc}, vm_ms #{r["vm_ms"]}, pumps #{r["pumps"]}, #{r["result_bytes"]} result bytes")
    r
  end

  # --- comparing the two -----------------------------------------------------------

  defp compare(tdir, idir) do
    t = read_report(tdir)
    i = read_report(idir)
    tout = File.read(Path.join(tdir, "out.f32"))
    iout = File.read(Path.join(idir, "out.f32"))
    flipped = with {:ok, b} <- iout, true <- byte_size(b) > 0, do: {:ok, flip_byte(b, div(byte_size(b), 2))}, else: (_ -> iout)
    nan = with {:ok, b} <- tout, true <- byte_size(b) >= 4, do: {:ok, <<0x7FC00000::little-32>> <> binary_part(b, 4, byte_size(b) - 4)}, else: (_ -> tout)
    [
      check("both runs ran the same ELF (sha256 #{t["elf_sha256"]})", same(t, i, "elf_sha256")),
      check("both runs loaded the same program (hash #{t["hash"]})", same(t, i, "hash")),
      check("the translated path ran: translated, and at least #{@min_speedup}x the interpreter's speed", speedup(t, i)),
      check("control: the interpreted run is refused by the same check", refused(speedup(i, i))),
      check("the translated outputs equal the interpreter's, byte for byte", equal(tout, iout)),
      check("control: the interpreter's outputs with one byte changed are refused", refused(equal(tout, flipped))),
      check("the outputs are the expected ones (sha256 #{String.slice(@expected, 0, 8)})", expected(tout)),
      check("control: the expected check refuses the interpreter's outputs with one byte changed", refused(expected(flipped))),
      check("the outputs are #{@result_floats} finite floats with every box width and height above zero", sane(tout)),
      check("control: outputs with a NaN planted are refused", refused(sane(nan)))
    ]
  end

  defp loads_translated(%{rc: 0} = r, hash) do
    cond do
      r["hash"] != hash -> {:error, "hash #{r["hash"]}, not #{hash}"}
      r["translated"] != "yes" -> {:error, "is_binary_translated() is false"}
      true -> :ok
    end
  end

  defp loads_translated(r, _), do: {:error, "godot exited #{r.rc}: #{r["FAIL"] || "no FAIL line"}"}

  defp ran(%{rc: 0} = r, hash, translated) do
    cond do
      r["hash"] != hash -> {:error, "hash #{r["hash"]}, not #{hash}"}
      r["translated"] != translated or r["translated_after"] != translated ->
        {:error, "translated #{r["translated"]} before and #{r["translated_after"]} after, not #{translated}"}
      r["result_bytes"] != to_string(@result_floats * 4) -> {:error, "#{r["result_bytes"]} result bytes"}
      true -> :ok
    end
  end

  defp ran(r, _, _), do: {:error, "godot exited #{r.rc}: #{r["FAIL"] || "no FAIL line"}"}

  defp speedup(t, i) do
    with {tm, ""} <- Integer.parse(t["vm_ms"] || ""), {im, ""} <- Integer.parse(i["vm_ms"] || "") do
      ratio = im / max(tm, 1)
      cond do
        t["translated"] != "yes" or t["translated_after"] != "yes" -> {:error, "is_binary_translated() is false"}
        ratio < @min_speedup -> {:error, "#{im} ms interpreted against #{tm} ms, #{Float.round(ratio, 2)}x"}
        true -> say("  #{tm} ms translated, #{im} ms interpreted: #{Float.round(ratio, 2)}x")
      end
    else
      _ -> {:error, "a report has no vm_ms"}
    end
  end

  defp same(t, i, key) do
    if t[key] != nil and t[key] == i[key], do: :ok, else: {:error, "#{inspect(t[key])} and #{inspect(i[key])}"}
  end

  defp equal({:ok, a}, {:ok, a}) when byte_size(a) > 0, do: say("  sha256 #{sha256(a)}")
  defp equal({:ok, a}, {:ok, b}), do: {:error, "#{byte_size(a)} and #{byte_size(b)} bytes, first difference at #{first_diff(a, b)}"}
  defp equal(a, b), do: {:error, "an output is missing: #{inspect(elem(a, 0))}, #{inspect(elem(b, 0))}"}

  defp expected({:ok, b}) do
    if sha256(b) == @expected, do: :ok, else: {:error, "sha256 #{sha256(b)}"}
  end

  defp expected(_), do: {:error, "no output"}

  # Box width and height are exp(delta) times the reference's, so positive but not bounded by 1.
  defp sane({:ok, b}) when byte_size(b) == @result_floats * 4 do
    bad = for <<bits::little-32 <- b>>, Bitwise.band(Bitwise.bsr(bits, 23), 0xFF) == 0xFF, reduce: 0, do: (n -> n + 1)
    sizes = for <<_::binary-size(8), w::float-little-32, h::float-little-32 <- binary_part(b, 0, 1600)>>, do: min(w, h)
    cond do
      bad > 0 -> {:error, "#{bad} non-finite floats"}
      Enum.any?(sizes, &(&1 <= 0)) -> {:error, "a box with a width or height at or below zero"}
      true -> :ok
    end
  end

  defp sane({:ok, b}), do: {:error, "#{byte_size(b)} bytes, not #{@result_floats * 4}"}
  defp sane(_), do: {:error, "no output"}

  # --- what the runs need ----------------------------------------------------------

  defp host do
    case {:os.type(), to_string(:erlang.system_info(:system_architecture))} do
      {{:unix, :linux}, "x86_64" <> _} -> :linux
      {{:unix, :darwin}, _} -> :macos
      other -> fail("no pinned double engine and addon for #{inspect(other)}")
    end
  end

  defp setup(host, work) do
    cache = Path.join(work, "cache")
    {etag, ename, esha} = @engine[host]
    engine = fetch("#{@godot_build}/#{etag}/#{ename}", Path.join(cache, ename), esha)
    File.chmod!(engine, 0o755)
    {atag, aname, asha} = @addon[host]
    addon = fetch("#{@godot_build}/#{atag}/#{aname}", Path.join(cache, aname), asha)
    File.cp!(addon, Path.join(@project, "addons/godot_sandbox/bin/#{aname}"))
    File.mkdir_p!(Path.join(@project, ".godot"))
    File.write!(Path.join(@project, ".godot/extension_list.cfg"), "res://addons/godot_sandbox/bin/godot-riscv.gdextension\n")
    models = Path.join(cache, "models")
    for {name, sha} <- @models, do: fetch("#{@models_url}/#{name}", Path.join(models, name), sha)
    frame = Path.join(work, "frame.f32")
    File.write!(frame, frame())
    say("frame: #{frame}, sha256 #{sha256(File.read!(frame))}")
    {suffix, flags} = if host == :macos, do: {".dylib", ~w(-dynamiclib)}, else: {".so", ~w(-fPIC)}
    xvfb = host == :linux and System.get_env("DISPLAY") in [nil, ""] and System.find_executable("xvfb-run")
    %{engine: engine, models: models, frame: frame, suffix: suffix, flags: flags, host: host, xvfb: xvfb,
      cc: System.get_env("BINTR_CC") || "clang"}
  end

  # A 312 x 312 RGB frame, planar and ImageNet-normalised as the guest takes it: an ellipse on a gradient, in integer
  # arithmetic and one correctly rounded division, so every host writes the same bytes.
  defp frame do
    mean = {0.485, 0.456, 0.406}
    std = {0.229, 0.224, 0.225}
    figure = {200, 120, 90}
    for c <- 0..2, y <- 0..311, x <- 0..311, into: <<>> do
      dx = x - 156
      dy = y - 170
      v = if dx * dx * 14_400 + dy * dy * 3_600 <= 3_600 * 14_400, do: elem(figure, c), else: div((y + c * 40) * 255, 391)
      <<(v / 255 - elem(mean, c)) / elem(std, c)::float-32-little>>
    end
  end

  defp fetch(url, path, sha) do
    unless File.exists?(path) and sha256(File.read!(path)) == sha do
      File.mkdir_p!(Path.dirname(path))
      part = path <> ".part"
      say("$ curl #{url}")
      {out, rc} = System.cmd("curl", ["-fsSL", "--retry", "3", "-o", part, url], stderr_to_stdout: true)
      if rc != 0, do: fail("curl exited #{rc}: #{out}")
      got = sha256(File.read!(part))
      if got != sha, do: fail("#{Path.basename(path)}: sha256 #{got}, pinned #{sha}")
      File.rename!(part, path)
    end
    say("#{Path.basename(path)}: sha256 #{sha}")
    path
  end

  defp compile(env, c, lib) do
    args = ~w(-O2 -s -std=c99 -shared -x c -fexceptions -fvisibility=hidden -fomit-frame-pointer) ++ env.flags ++ [c, "-o", lib]
    say("$ #{env.cc} #{Enum.join(args, " ")}")
    t0 = System.monotonic_time(:millisecond)
    {out, rc} = System.cmd(env.cc, args, stderr_to_stdout: true)
    if rc != 0, do: fail("#{env.cc} exited #{rc}: #{out}")
    say("translation compiled in #{System.monotonic_time(:millisecond) - t0} ms: #{Path.basename(lib)}")
  end

  # Godot under a wall clock; the window goes to xvfb on a Linux runner without a display.
  defp godot(env, kind, args, extra_env \\ []) do
    driver = if env.host == :linux, do: ["--rendering-driver", "opengl3"], else: []
    cmd = [env.engine, "--audio-driver", "Dummy"] ++ driver ++ ["--path", @project, "--script", "run_seg.gd", "--"] ++ args
    cmd = if env.xvfb, do: [env.xvfb, "-a" | cmd], else: cmd
    say("$ #{Enum.join(cmd, " ")}")
    port = Port.open({:spawn_executable, hd(cmd)}, [:binary, :exit_status, :stderr_to_stdout, args: tl(cmd),
                                                   env: Enum.map(extra_env, fn {k, v} -> {~c"#{k}", ~c"#{v}"} end)])
    deadline = System.monotonic_time(:millisecond) + @wall_s[kind] * 1000
    {out, rc} = collect(port, deadline, [])
    lines = for l <- String.split(out, "\n"), [k, v] <- [String.split(String.trim(l), " ", parts: 2)], do: {k, v}
    Map.merge(Map.new(lines), %{out: out, rc: rc})
  end

  defp collect(port, deadline, acc) do
    receive do
      {^port, {:data, d}} -> collect(port, deadline, [acc, d])
      {^port, {:exit_status, rc}} -> {IO.iodata_to_binary(acc), rc}
    after
      max(deadline - System.monotonic_time(:millisecond), 0) ->
        {:os_pid, pid} = Port.info(port, :os_pid)
        System.cmd("kill", ["-9", to_string(pid)])
        {IO.iodata_to_binary(acc) <> "\nFAIL past the wall clock\n", 124}
    end
  end

  # The byte at the middle of .text, low bit flipped: the code, and so the translation hash, change.
  defp flip_text(elf) do
    <<_::binary-size(0x28), shoff::little-64, _::binary-size(10), entsize::little-16, n::little-16, shstrndx::little-16, _::binary>> = elf
    shdr = fn i -> <<name::little-32, _::binary-size(20), off::little-64, size::little-64, _::binary>> = binary_part(elf, shoff + i * entsize, entsize); {name, off, size} end
    {_, stroff, _} = shdr.(shstrndx)
    {_, off, size} =
      Enum.find_value(0..(n - 1), fn i ->
        {name, _, _} = s = shdr.(i)
        if binary_part(elf, stroff + name, 6) == ".text" <> <<0>>, do: s
      end) || fail("no .text section")
    flip_byte(elf, off + div(size, 2))
  end

  defp flip_byte(b, at), do: binary_part(b, 0, at) <> <<Bitwise.bxor(:binary.at(b, at), 1)>> <> binary_part(b, at + 1, byte_size(b) - at - 1)

  defp first_diff(a, b) do
    Enum.find(0..(min(byte_size(a), byte_size(b)) - 1)//1, min(byte_size(a), byte_size(b)), &(:binary.at(a, &1) != :binary.at(b, &1)))
  end

  defp read_report(dir) do
    case File.read(Path.join(dir, "report.txt")) do
      {:ok, s} -> Map.new(for l <- String.split(s, "\n", trim: true), [k, v] <- [String.split(l, " ", parts: 2)], do: {k, v})
      _ -> %{}
    end
  end

  defp reset(dir) do
    File.rm_rf!(dir)
    File.mkdir_p!(dir)
  end

  defp refused(:ok), do: {:error, "accepted"}
  defp refused({:error, why}), do: say("  refused: #{why}")

  defp check(name, :ok), do: (say("PASS #{name}"); :ok)
  defp check(name, {:error, why}), do: (say("FAIL #{name}: #{why}"); :fail)

  defp sha256(b), do: Base.encode16(:crypto.hash(:sha256, b), case: :lower)
  defp say(msg), do: (IO.puts("== #{msg}"); :ok)

  defp fail(msg) do
    IO.puts(:stderr, "check_bintr: #{msg}")
    System.halt(1)
  end
end

CheckBintr.main(System.argv())
