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

function Find-PythonLauncher {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) { return $launcher.Source }

    $candidates = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\Python\Launcher\py.exe'),
        (Join-Path $env:WINDIR 'py.exe')
    )
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { return $candidate }
    }
    return $null
}

function Test-Python312Runtime {
    param([string]$Launcher)
    try {
        & $Launcher -3.12 -c 'import sys; print(sys.executable)' 2>$null | Out-Null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

function Install-Python312Runtime {
    $installerUrl = 'https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe'
    $installerPath = Join-Path ([IO.Path]::GetTempPath()) ('python-3.12.10-' + [guid]::NewGuid().ToString('N') + '.exe')
    $pythonHome = Join-Path $env:LOCALAPPDATA 'Programs\Python\Python312'
    try {
        Write-Host 'Downloading the official Python 3.12.10 installer from python.org...'
        Invoke-WebRequest -Uri $installerUrl -OutFile $installerPath -UseBasicParsing
        $signature = Get-AuthenticodeSignature -FilePath $installerPath
        if ($signature.Status -ne 'Valid' -or $signature.SignerCertificate.Subject -notmatch 'Python Software Foundation') {
            throw 'The downloaded Python installer signature is invalid or is not from the Python Software Foundation.'
        }

        $arguments = "/quiet InstallAllUsers=0 PrependPath=0 Include_launcher=1 InstallLauncherAllUsers=0 Include_pip=1 Include_test=0 TargetDir=`"$pythonHome`""
        $process = Start-Process -FilePath $installerPath -ArgumentList $arguments -Wait -PassThru
        if ($process.ExitCode -notin @(0, 3010)) { throw "Python installation failed with exit code $($process.ExitCode)." }

        $pythonPath = Join-Path $pythonHome 'python.exe'
        if (-not (Test-Path $pythonPath)) { throw 'Python installer completed, but python.exe was not found in the expected location.' }
        return $pythonPath
    } finally {
        if (Test-Path $installerPath) { Remove-Item -LiteralPath $installerPath -Force }
    }
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
$pythonArguments = @()
if (-not (Test-Path $venvPython)) {
    $python = Find-PythonLauncher
    $hasPython312 = $python -and (Test-Python312Runtime $python)
    if ($hasPython312) { $pythonArguments = @('-3.12') }
    if (-not $hasPython312) {
        Write-Host 'Python 3.12 runtime not found. Installing it without requiring winget...'
        $python = Install-Python312Runtime
    }
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
    if (-not (Test-Path $venvPython)) { Invoke-Native $python ($pythonArguments + @('-m','venv',(Join-Path $InstallDir '.venv'))) }
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
