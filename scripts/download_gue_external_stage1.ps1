$ErrorActionPreference = "Stop"

$root = Join-Path $PSScriptRoot "..\data\gue_external"
$files = @(
    @{ Task = "human_tf_0"; Split = "train"; Name = "train.csv"; Url = "https://huggingface.co/datasets/leannmlindsey/GUE/resolve/main/GUE/human_tf_0/train.csv?download=true" },
    @{ Task = "human_tf_0"; Split = "validation"; Name = "dev.csv"; Url = "https://huggingface.co/datasets/leannmlindsey/GUE/resolve/main/GUE/human_tf_0/dev.csv?download=true" },
    @{ Task = "human_tf_0"; Split = "test"; Name = "test.csv"; Url = "https://huggingface.co/datasets/leannmlindsey/GUE/resolve/main/GUE/human_tf_0/test.csv?download=true" },
    @{ Task = "mouse_0"; Split = "train"; Name = "train.csv"; Url = "https://huggingface.co/datasets/leannmlindsey/GUE/resolve/main/GUE/mouse_0/train.csv?download=true" },
    @{ Task = "mouse_0"; Split = "validation"; Name = "dev.csv"; Url = "https://huggingface.co/datasets/leannmlindsey/GUE/resolve/main/GUE/mouse_0/dev.csv?download=true" },
    @{ Task = "mouse_0"; Split = "test"; Name = "test.csv"; Url = "https://huggingface.co/datasets/leannmlindsey/GUE/resolve/main/GUE/mouse_0/test.csv?download=true" }
)

foreach ($file in $files) {
    $directory = Join-Path $root $file.Task
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
    $destination = Join-Path $directory $file.Name
    if (Test-Path -LiteralPath $destination) {
        throw "Refusing to overwrite existing GUE asset: $destination"
    }
    Invoke-WebRequest -Uri $file.Url -OutFile $destination
    $hash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
    $size = (Get-Item -LiteralPath $destination).Length
    Write-Output "$($file.Task),$($file.Split),$size,$hash,$destination"
}
