[CmdletBinding()]
param(
    [string]$RepositoryOwner = 'opoeta',
    [string]$RepositoryName = 'tunet',
    [string]$Branch = 'windows',
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA 'Programs\TuNet'),
    [switch]$CpuOnly
)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Invoke-Native {
    param([string]$File, [string[]]$Arguments)
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Falha ($LASTEXITCODE): $File $($Arguments -join ' ')" }
}

if ($env:OS -ne 'Windows_NT') { throw 'Este instalador requer Windows.' }
$venvPython = Join-Path $InstallDir '.venv\Scripts\python.exe'
$isUpdate = (Test-Path (Join-Path $InstallDir 'ui_app.py')) -and (Test-Path $venvPython)
if ($isUpdate) {
    Write-Host "Existing TuNet installation detected. Updating files in: $InstallDir"
} else {
    Write-Host "No complete TuNet installation detected. Preparing a new installation in: $InstallDir"
}

$python = $null
if (-not (Test-Path $venvPython)) {
    if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
        if (Get-Command winget -ErrorAction SilentlyContinue) {
            Write-Host 'Installing Python 3.12 for the current user...'
            Invoke-Native 'winget' @('install','--id','Python.Python.3.12','--exact','--scope','user','--accept-package-agreements','--accept-source-agreements')
        } else { throw 'Install Python 3.12 with the py launcher and run this script again.' }
    }
    $python = (Get-Command py).Source
    & $python -3.12 -c 'import sys; print(sys.executable)' 2>$null
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 was not found. Install it and run this script again.' }
}

$work = Join-Path ([IO.Path]::GetTempPath()) ('tunet-install-' + [guid]::NewGuid().ToString('N'))
$sourceRoot = Join-Path $work 'source'
New-Item -ItemType Directory -Path $sourceRoot -Force | Out-Null
try {
    $apiHeaders = @{ 'Accept' = 'application/vnd.github+json'; 'User-Agent' = 'TuNet-Windows-Installer' }
    $treeUrl = "https://api.github.com/repos/$RepositoryOwner/$RepositoryName/git/trees/$Branch`?recursive=1"
    Write-Host "Lendo a árvore $RepositoryOwner/$RepositoryName ($Branch) no GitHub..."
    $tree = Invoke-RestMethod -Uri $treeUrl -Headers $apiHeaders
    if (-not $tree.tree) { throw 'Não foi possível localizar a branch no GitHub.' }
    if ($tree.truncated) { throw 'A árvore retornada pelo GitHub foi truncada; instalação interrompida para evitar arquivos incompletos.' }
    $files = @($tree.tree | Where-Object { $_.type -eq 'blob' })
    $requiredFiles = @('ui_app.py','train.py','base/base.yaml')
    foreach ($required in $requiredFiles) {
        if ($required -notin $files.path) { throw "Arquivo obrigatório ausente na branch: $required" }
    }

    $branchPath = [uri]::EscapeDataString($Branch)
    foreach ($entry in $files) {
        $relative = [string]$entry.path
        if ([IO.Path]::IsPathRooted($relative) -or ($relative.Split('/') -contains '..')) {
            throw "Caminho inválido recebido do GitHub: $relative"
        }
        $urlPath = (($relative -split '/') | ForEach-Object { [uri]::EscapeDataString($_) }) -join '/'
        $destination = Join-Path $sourceRoot ($relative.Replace('/', [IO.Path]::DirectorySeparatorChar))
        $fullDestination = [IO.Path]::GetFullPath($destination)
        $rootPrefix = [IO.Path]::GetFullPath($sourceRoot).TrimEnd([IO.Path]::DirectorySeparatorChar) + [IO.Path]::DirectorySeparatorChar
        if (-not $fullDestination.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Destino fora da pasta temporária: $relative"
        }
        New-Item -ItemType Directory -Path (Split-Path -Parent $fullDestination) -Force | Out-Null
        $rawUrl = "https://raw.githubusercontent.com/$RepositoryOwner/$RepositoryName/$branchPath/$urlPath"
        Invoke-WebRequest -Uri $rawUrl -OutFile $fullDestination -UseBasicParsing -Headers @{ 'User-Agent' = 'TuNet-Windows-Installer' }
    }
    Write-Host ("Baixados {0} arquivos individualmente do GitHub." -f $files.Count)

    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    Get-ChildItem -LiteralPath $sourceRoot -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $InstallDir -Recurse -Force
    }
    if (-not (Test-Path $venvPython)) { Invoke-Native $python @('-3.12','-m','venv',(Join-Path $InstallDir '.venv')) }
    Invoke-Native $venvPython @('-m','pip','install','--upgrade','pip')
    if ($CpuOnly) {
        Invoke-Native $venvPython @('-m','pip','install','torch','torchvision','--index-url','https://download.pytorch.org/whl/cpu')
    } else {
        Invoke-Native $venvPython @('-m','pip','install','torch','torchvision','--index-url','https://download.pytorch.org/whl/cu132')
    }
    Invoke-Native $venvPython @('-m','pip','install','onnx','pyyaml','lpips','onnxruntime','Pillow','albumentations','PySide6')

    $shortcutDir = Join-Path $env:APPDATA 'Microsoft\Windows\Start Menu\Programs'
    New-Item -ItemType Directory -Path $shortcutDir -Force | Out-Null
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut((Join-Path $shortcutDir 'TuNet.lnk'))
    $shortcut.TargetPath = Join-Path $InstallDir '.venv\Scripts\pythonw.exe'
    $shortcut.Arguments = ('"' + (Join-Path $InstallDir 'ui_app.py') + '"')
    $shortcut.WorkingDirectory = $InstallDir
    $shortcut.Description = 'TuNet — treinamento e inferência de imagens'
    $shortcut.Save()
    if ($isUpdate) {
        Write-Host "TuNet update completed: $InstallDir"
        Write-Host 'The existing virtual environment was reused. Open TuNet from the Start menu.'
    } else {
        Write-Host "New TuNet installation completed: $InstallDir"
        Write-Host 'Open TuNet from the Start menu. Checkpoints are saved to the folder selected in the interface.'
    }
} finally {
    if (Test-Path $work) { Remove-Item -LiteralPath $work -Recurse -Force }
}
