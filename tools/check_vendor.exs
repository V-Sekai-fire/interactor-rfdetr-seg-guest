# vendor/rf-detr-ggml/src against interactor-rf-detr-ggml at the commit its CITATION.cff names, byte for byte.
#   elixir tools/check_vendor.exs [--source=<interactor-rf-detr-ggml checkout>]
defmodule CheckVendor do
  @root Path.expand("..", __DIR__)
  @vendor Path.join(@root, "vendor/rf-detr-ggml")
  @files for s <- ~w(ops backbone projector deform_attn decoder keypoints segmentation), x <- ~w(.cpp .h), do: "src/#{s}#{x}"

  def main(argv) do
    {kv, _, _} = OptionParser.parse(argv, strict: [source: :string])
    cff = File.read!(Path.join(@vendor, "CITATION.cff"))
    [_, commit] = Regex.run(~r/^commit: ([0-9a-f]{40})$/m, cff) || fail("CITATION.cff names no commit")
    [_, url] = Regex.run(~r/^repository-code: (\S+)$/m, cff) || fail("CITATION.cff names no repository-code")
    src = kv[:source] || clone(url, commit)
    upstream = Map.new(@files, &{&1, show(src, commit, &1)})
    local = Map.new(@files, &{&1, File.read(Path.join(@vendor, &1))})
    extra = (Path.wildcard(Path.join(@vendor, "src/*")) |> Enum.map(&Path.relative_to(&1, @vendor))) -- @files
    {first, last} = {hd(@files), List.last(@files)}
    flipped = Map.update!(local, first, fn {:ok, b} -> {:ok, flip(b)}; e -> e end)
    dropped = Map.put(local, last, {:error, :enoent})
    results = [
      check("#{length(@files)} files equal #{url} at #{String.slice(commit, 0, 7)}", same(upstream, local)),
      check("no file beside them that the source does not hold", if(extra == [], do: :ok, else: {:error, inspect(extra)})),
      check("every vendored #include resolves to a vendored file or a build dependency", includes(local)),
      check("control: #{first} with one byte flipped is refused", refused(same(upstream, flipped))),
      check("control: #{last} missing is refused", refused(same(upstream, dropped))),
      check("control: an include of the source's dataset.h is refused",
        refused(includes(Map.put(local, "src/extra.cpp", {:ok, "#include \"dataset.h\"\n"}))))
    ]
    failed = Enum.count(results, &(&1 != :ok))
    IO.puts("== #{length(results) - failed} of #{length(results)} checks pass")
    if failed > 0, do: System.halt(1)
  end

  defp clone(url, commit) do
    dir = Path.join(System.tmp_dir!(), "rf-detr-ggml-#{commit}")
    unless File.dir?(Path.join(dir, ".git")) do
      git(["init", "-q", dir])
      git(["-C", dir, "fetch", "-q", "--depth=1", url, commit])
    end
    dir
  end

  defp show(src, commit, file) do
    case System.cmd("git", ["-C", src, "show", "#{commit}:#{file}"], stderr_to_stdout: true) do
      {b, 0} -> {:ok, b}
      {out, _} -> {:error, String.trim(out)}
    end
  end

  defp same(upstream, local) do
    bad = for f <- @files, upstream[f] != local[f] or not match?({:ok, _}, local[f]), do: f
    if bad == [], do: :ok, else: {:error, "differ: #{Enum.join(bad, ", ")}"}
  end

  # Quoted includes either name a vendored header or one the guest build supplies (ggml, ggml-rd, the runtime).
  @provided ~w(ggml.h ggml-alloc.h ggml-backend.h gguf.h ggml-rd.h)
  defp includes(local) do
    have = for {f, {:ok, _}} <- local, do: Path.basename(f)
    bad =
      for {f, {:ok, b}} <- local, [_, inc] <- Regex.scan(~r/^#include "([^"]+)"/m, b),
          inc not in have and inc not in @provided, do: "#{f}: #{inc}"
    if bad == [], do: :ok, else: {:error, Enum.join(bad, "; ")}
  end

  defp flip(<<c, rest::binary>>), do: <<Bitwise.bxor(c, 1), rest::binary>>

  defp git(args) do
    {out, rc} = System.cmd("git", args, stderr_to_stdout: true)
    if rc != 0, do: fail("git #{Enum.join(args, " ")}: #{out}")
  end

  defp refused(:ok), do: {:error, "accepted"}
  defp refused({:error, why}), do: (IO.puts("==   refused: #{why}"); :ok)

  defp check(name, :ok), do: (IO.puts("PASS #{name}"); :ok)
  defp check(name, {:error, why}), do: (IO.puts("FAIL #{name}: #{why}"); :fail)

  defp fail(msg) do
    IO.puts(:stderr, "check_vendor: #{msg}")
    System.halt(1)
  end
end

CheckVendor.main(System.argv())
