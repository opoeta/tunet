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
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host 'Instalando Python 3.12 no escopo do usuário...'
        Invoke-Native 'winget' @('install','--id','Python.Python.3.12','--exact','--scope','user','--accept-package-agreements','--accept-source-agreements')
    } else { throw 'Instale Python 3.12 (com o launcher py) e execute novamente.' }
}
$python = (Get-Command py).Source
$pythonCheck = & $python -3.12 -c 'import sys; print(sys.executable)' 2>$null
if ($LASTEXITCODE -ne 0) { throw 'Python 3.12 não foi encontrado. Instale-o e execute novamente.' }

$work = Join-Path ([IO.Path]::GetTempPath()) ('tunet-install-' + [guid]::NewGuid().ToString('N'))
$archive = Join-Path $work 'source.zip'
$extract = Join-Path $work 'source'
New-Item -ItemType Directory -Path $extract -Force | Out-Null
$archiveUrl = "https://github.com/$RepositoryOwner/$RepositoryName/archive/refs/heads/$Branch.zip"
try {
    Write-Host "Baixando $RepositoryOwner/$RepositoryName ($Branch)..."
    Invoke-WebRequest -Uri $archiveUrl -OutFile $archive
    Expand-Archive -LiteralPath $archive -DestinationPath $extract
    $source = Get-ChildItem -LiteralPath $extract -Directory | Select-Object -First 1
    if (-not $source -or -not (Test-Path (Join-Path $source.FullName 'ui_app.py')) -or -not (Test-Path (Join-Path $source.FullName 'train.py'))) {
        throw 'O arquivo do GitHub não contém os arquivos esperados. Confira repositório e branch.'
    }
    New-Item -ItemType Directory -Path $InstallDir -Force | Out-Null
    Get-ChildItem -LiteralPath $source.FullName -Force | ForEach-Object {
        Copy-Item -LiteralPath $_.FullName -Destination $InstallDir -Recurse -Force
    }
    $venvPython = Join-Path $InstallDir '.venv\Scripts\python.exe'
    if (-not (Test-Path $venvPython)) { Invoke-Native $python @('-3.12','-m','venv',(Join-Path $InstallDir '.venv')) }
    $venvPython = Join-Path $InstallDir '.venv\Scripts\python.exe'
    Invoke-Native $venvPython @('-m','pip','install','--upgrade','pip')
    if ($CpuOnly) {
        Invoke-Native $venvPython @('-m','pip','install','torch','torchvision','--index-url','https://download.pytorch.org/whl/cpu')
    } else {
        # O índice oficial seleciona as wheels CUDA estáveis mais recentes compatíveis com Windows.
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
    Write-Host "Instalação concluída: $InstallDir"
    Write-Host 'Abra TuNet pelo menu Iniciar. Os checkpoints ficam na pasta escolhida na interface.'
} finally {
    if (Test-Path $work) { Remove-Item -LiteralPath $work -Recurse -Force }
}
