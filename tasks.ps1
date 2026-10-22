# Windows equivalent of the Makefile.  Usage:  .\tasks.ps1 <target> [arg]
#   .\tasks.ps1 install | up | down | recorder-api | recorder-web | check data\raw\<session> | test
param([Parameter(Position = 0)][string]$Target = "help", [Parameter(Position = 1)][string]$Arg = "")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Py = ".\.venv\Scripts\python.exe"

switch ($Target) {
    "venv" { python -m venv .venv }
    "install" {
        & $Py -m pip install --upgrade pip
        & $Py -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
        & $Py -m pip install -r requirements.txt
    }
    "up" { docker compose up -d; docker compose ps }
    "down" { docker compose down }
    "ps" { docker compose ps }
    "questions" { & $Py -m tools.make_questions }
    "recorder-api" { & $Py -m uvicorn recorder.backend.main:app --host 0.0.0.0 --port 8010 --reload --reload-dir recorder --reload-dir tools }
    "recorder-web-install" { Push-Location recorder\web; npm install; Pop-Location }
    "recorder-web" { Push-Location recorder\web; npm run dev; Pop-Location }
    "check" { & $Py -m tools.check_session $Arg }
    "test" { & $Py -m pytest -q tests }
    default {
        Write-Output "targets: venv install up down ps questions recorder-api recorder-web-install recorder-web check <dir> test"
    }
}
