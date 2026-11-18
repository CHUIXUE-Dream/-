param([string]$Python='',[switch]$FullRefit,[switch]$Figures)
$ErrorActionPreference='Stop'
if (-not $Python) {
 $bundled=Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
 if (Test-Path -LiteralPath $bundled) {$Python=$bundled} else {$Python=(Get-Command python -ErrorAction Stop).Source}
}
$args0=@((Join-Path $PSScriptRoot '运行.py'))
if($FullRefit){$args0+='--full-refit'}
if($Figures){$args0+='--figures'}
& $Python @args0
if($LASTEXITCODE -ne 0){throw 'Reproduction failed'}
