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
    "labelstudio" { docker compose --profile labelling up -d labelstudio; Write-Output "Label Studio: http://localhost:8090 (login in .env)" }
    "labelstudio-import" { & $Py -m tools.import_to_labelstudio --push }
    "labels" {
        & $Py -m tools.export_labels --pull
        & $Py -m tools.synthetic_to_labels
        & $Py -m tools.agreement --labels data/processed/labels/recorded
        & $Py -m tools.agreement --labels data\processed\labels\synthetic
        & $Py -m tools.make_splits
        & $Py -m tools.dataset_stats
    }
    "datasets" { & $Py -m tools.datasets.download_all }
    "splits" { & $Py -m tools.make_splits }
    "data" { & $Py -m ml.common.make_data }
    "eda" { & $Py notebooks\make_eda.py }
    "stats" { & $Py -m tools.dataset_stats }
    # ---- Stages 4-8: train / evaluate (one at a time keeps RAM use low)
    "train-all" { & $Py -m ml.train_all $Arg }
    "export" { & $Py -m ml.export.export_onnx; & $Py -m ml.export.benchmark }
    "retrain-live" { & $Py -m ml.live_retrain }
    "demo-webcam" { & $Py -m ml.vision.demo_webcam }
    "demo-mic" { & $Py -m ml.speech.demo_mic }
    # ---- Stage 9: product
    "api" { & $Py -m uvicorn backend.main:app --host 0.0.0.0 --port 8020 }
    "workers" { & $Py -m backend.workers.run_all }
    "web-install" { Push-Location frontend; npm install; Pop-Location }
    "web" { Push-Location frontend; npm run dev; Pop-Location }
    "start-all" {
        docker compose up -d postgres redis livekit caddy
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$PSScriptRoot'; .\tasks.ps1 api"
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$PSScriptRoot'; .\tasks.ps1 workers"
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$PSScriptRoot'; .\tasks.ps1 web"
        Write-Output "MBAS: http://localhost:3020 (this laptop)  |  https://<LAN_IP>:8444 (second laptop)"
    }
    # ---- Stage 10: evaluation + documents
    "reports" { & $Py -m tools.model_report; & $Py -m tools.bias_check; & $Py -m tools.user_study --demo }
    default {
        Write-Output "targets: venv install up down ps questions recorder-api recorder-web-install recorder-web check <dir> test labelstudio labelstudio-import labels datasets splits stats data eda train-all export retrain-live demo-webcam demo-mic api workers web-install web start-all reports"
    }
}
