param(
  [string[]]$Sources = @()
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

if ($Sources.Count -eq 0) {
  $Sources = @(Get-ChildItem (Join-Path $projectRoot "artifacts") -Filter *.md | ForEach-Object {
    "artifacts/$($_.Name)"
  })
}

$pandocCommand = Get-Command pandoc -ErrorAction SilentlyContinue
$pandoc = $pandocCommand?.Source
if (-not $pandoc) {
  $pandoc = Get-ChildItem (Join-Path $projectRoot ".tools/pandoc") -Recurse -Filter pandoc.exe -ErrorAction SilentlyContinue |
    Select-Object -First 1 -ExpandProperty FullName
}
if (-not $pandoc) { throw "Pandoc was not found. Install it on PATH or place it under .tools/pandoc/." }

$tectonicCommand = Get-Command tectonic -ErrorAction SilentlyContinue
$tectonic = $tectonicCommand?.Source
if (-not $tectonic) {
  $tectonic = Join-Path $projectRoot ".tools/tectonic/tectonic.exe"
}
if (-not (Test-Path -LiteralPath $tectonic)) { throw "Tectonic was not found. Install it on PATH or place it at .tools/tectonic/tectonic.exe." }
$header = Join-Path $projectRoot "latex/preamble.tex"
$texOutput = Join-Path $projectRoot "output/tex"
$pdfOutput = Join-Path $projectRoot "output/pdf"
$scratch = Join-Path $projectRoot "tmp/pdfs"

New-Item -ItemType Directory -Force -Path $texOutput, $pdfOutput, $scratch | Out-Null

foreach ($sourceRelative in $Sources) {
  $source = Join-Path $projectRoot $sourceRelative
  if (-not (Test-Path -LiteralPath $source)) { throw "Artifact source not found: $sourceRelative" }
  $stem = [IO.Path]::GetFileNameWithoutExtension($source)
  $lines = Get-Content -LiteralPath $source -Encoding UTF8

  $titleIndex = [Array]::FindIndex($lines, [Predicate[string]] { param($line) $line -match '^#\s+' })
  if ($titleIndex -lt 0) { throw "No level-one title found in $sourceRelative" }
  $title = $lines[$titleIndex] -replace '^#\s+', ''

  $nextContentIndex = $titleIndex + 1
  while ($nextContentIndex -lt $lines.Count -and [string]::IsNullOrWhiteSpace($lines[$nextContentIndex])) { $nextContentIndex++ }
  $subtitle = $null
  if ($nextContentIndex -lt $lines.Count -and $lines[$nextContentIndex] -match '^\*\*(.+)\*\*$') {
    $subtitle = $Matches[1]
    $bodyIndex = $nextContentIndex + 1
  } else {
    $bodyIndex = $nextContentIndex
  }
  while ($bodyIndex -lt $lines.Count -and ($lines[$bodyIndex] -match '^\s*$|^---\s*$')) { $bodyIndex++ }
  $bodyLines = $lines[$bodyIndex..($lines.Count - 1)]

  # Pandoc supplies the durable TOC. Remove a hand-authored link list when it
  # appears at the beginning so the rendered document does not contain two.
  if ($bodyLines.Count -gt 0 -and $bodyLines[0] -match '^## Table of Contents\s*$') {
    $tocEnd = [Array]::FindIndex($bodyLines, 1, [Predicate[string]] { param($line) $line -match '^---\s*$' })
    if ($tocEnd -gt 0) {
      $bodyLines = $bodyLines[($tocEnd + 1)..($bodyLines.Count - 1)]
      while ($bodyLines.Count -gt 0 -and [string]::IsNullOrWhiteSpace($bodyLines[0])) { $bodyLines = $bodyLines[1..($bodyLines.Count - 1)] }
    }
  }

  # The supplied artifacts already embed numbers in their heading text.
  # Strip those labels so LaTeX can own numbering without producing "2. 2."
  # headings. Front matter and named appendices remain unnumbered.
  $bodyLines = @($bodyLines | ForEach-Object {
    $line = $_
    if ($line -match '^(#{2,6})\s+(?:Chapter\s+)?\d+(?:\.\d+)*[.:]?\s+(.+)$') {
      $line = "$($Matches[1]) $($Matches[2])"
    }
    if ($line -match '^##\s+(How to Use This Document|Appendix\s+[A-Z]:.+)$') {
      $line = "$line {.unnumbered}"
    }
    $line
  })

  $body = $bodyLines -join [Environment]::NewLine

  # Latin Modern Mono does not contain every Unicode math glyph. Only code
  # spans/blocks are normalized; prose and displayed mathematics stay intact.
  $codeGlyphs = [ordered]@{
    "·" = "*"; "ŷ" = "y_hat"; ([string][char]0x0303) = "";
    "Σ" = "Sigma"; "λ" = "lambda"; "ρ" = "rho"; "τ" = "tau";
    "β" = "beta"; "θ" = "theta"; "π" = "pi"; "∈" = "in"; "ε" = "epsilon";
    "≥" = ">="; "≤" = "<="; "≠" = "!="; "ℝ" = "R";
    "°" = " deg"; "²" = "^2"; "⁴" = "^4"; "⁸" = "^8"; "½" = "1/2"; "×" = "*";
    "ᵢ" = "_i"; "ₓ" = "_x"; "ᵧ" = "_y"; "₁" = "_1"; "₂" = "_2"; "₃" = "_3"; "₄" = "_4";
    "ₘ" = "_m"; "ₙ" = "_n"; "→" = "->"; "⟹" = "=>"
  }
  $body = [regex]::Replace($body, '(?s)```.*?```|`[^`\r\n]+`', {
    param($match)
    $value = $match.Value
    foreach ($entry in $codeGlyphs.GetEnumerator()) { $value = $value.Replace($entry.Key, $entry.Value) }
    $value = $value.Replace("Π", "Pi")
    $value
  })

  # Convert remaining Unicode math symbols in prose to native LaTeX math.
  $mathGlyphs = [ordered]@{
    "ℝ" = '$\mathbb{R}$'; "λ" = '$\lambda$'; "β" = '$\beta$'; "θ" = '$\theta$'; "π" = '$\pi$';
    "≥" = '$\geq$'; "≤" = '$\leq$'; "≠" = '$\neq$';
    "ₓ" = '$_x$'; "ᵧ" = '$_y$'; "ᵢ" = '$_i$'; "₁" = '$_1$'; "₂" = '$_2$';
    "₃" = '$_3$'; "₄" = '$_4$'; "ₘ" = '$_m$'; "ₙ" = '$_n$'; "⁴" = '$^4$'; "⁸" = '$^8$'
  }
  foreach ($entry in $mathGlyphs.GetEnumerator()) { $body = $body.Replace($entry.Key, $entry.Value) }
  $normalizedMarkdown = Join-Path $scratch "$stem.md"
  [IO.File]::WriteAllText($normalizedMarkdown, $body, [Text.UTF8Encoding]::new($false))

  $texFile = Join-Path $texOutput "$stem.tex"
  $pandocArgs = @(
    $normalizedMarkdown,
    '--from', 'markdown+tex_math_dollars', '--to', 'latex', '--standalone', '--toc', '--number-sections',
    '--top-level-division', 'chapter', '--shift-heading-level-by=-1', '--include-in-header', $header,
    '--metadata', "title=$title", '--metadata', 'author=Research Folio',
    '--variable', 'documentclass=report', '--variable', 'papersize=a4', '--variable', 'fontsize=11pt',
    '--variable', 'geometry:margin=26mm', '--variable', 'colorlinks=true',
    '--variable', 'linkcolor=artifactAccent', '--variable', 'urlcolor=artifactAccent',
    '--output', $texFile
  )
  if ($subtitle) {
    $outputPosition = [Array]::IndexOf($pandocArgs, '--output')
    $pandocArgs = @($pandocArgs[0..($outputPosition - 1)] + @('--metadata', "subtitle=$subtitle") + $pandocArgs[$outputPosition..($pandocArgs.Count - 1)])
  }
  & $pandoc @pandocArgs
  if ($LASTEXITCODE -ne 0) { throw "Pandoc failed for $sourceRelative" }

  # Pandoc's default two-column longtable uses natural-width columns, which
  # can overflow on prose-heavy rows. Give the description column room to wrap.
  $latex = [IO.File]::ReadAllText($texFile)
  $latex = $latex.Replace('\begin{longtable}[]{@{}ll@{}}', '\begin{longtable}[]{@{}p{0.24\linewidth}p{0.70\linewidth}@{}}')
  $latex = $latex.Replace('\begin{longtable}[]{@{}lll@{}}', '\begin{longtable}[]{@{}p{0.20\linewidth}p{0.16\linewidth}p{0.56\linewidth}@{}}')
  [IO.File]::WriteAllText($texFile, $latex, [Text.UTF8Encoding]::new($false))

  & $tectonic --keep-logs --keep-intermediates --outdir $pdfOutput $texFile
  if ($LASTEXITCODE -ne 0) { throw "Tectonic failed for $sourceRelative" }
}
