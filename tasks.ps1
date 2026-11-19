# Windows equivalent of the Makefile.  Usage:  .\tasks.ps1 <target> [arg]
#   .\tasks.ps1 install | up | down | recorder-api | recorder-web | check data\raw\<session> | test
param([Parameter(Position = 0)][string]$Target = "help", [Parameter(Position = 1)][string]$Arg = "")

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Py = ".\.venv\Scripts\python.exe"

# LiveKit must announce an address browsers will send video to. Chrome never uses 127.0.0.1 for WebRTC, so when
# .env has no real LAN address, use this laptop's current Wi-Fi / Ethernet IPv4 (the interface with the default
# route). Re-detected on every start, so changing networks or laptops just works.
function Set-LanIp {
    $fromEnv = (Get-Content .env -ErrorAction SilentlyContinue | Where-Object { $_ -match '^MBAS_LAN_IP=' }) -replace '^MBAS_LAN_IP=', ''
    if ($fromEnv -and $fromEnv -notmatch '^127\.') { $env:MBAS_LAN_IP = $fromEnv.Trim(); return }
    $route = Get-NetRoute -DestinationPrefix '0.0.0.0/0' -ErrorAction SilentlyContinue | Sort-Object RouteMetric | Select-Object -First 1
    $ip = if ($route) { (Get-NetIPAddress -InterfaceIndex $route.InterfaceIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue).IPAddress | Select-Object -First 1 }
    if ($ip) { $env:MBAS_LAN_IP = $ip; Write-Output "LiveKit address: $ip (detected)" }
    else { Write-Warning "No network found: video calls in Chrome need a Wi-Fi / LAN connection (127.0.0.1 is not used by browsers)." }
}

switch ($Target) {
    "venv" { python -m venv .venv }
    "install" {
        & $Py -m pip install --upgrade pip
        & $Py -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
        & $Py -m pip install -r requirements.txt
    }
    "up" { Set-LanIp; docker compose up -d --force-recreate livekit caddy; docker compose up -d; docker compose ps }
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
    # These run for hours and their libraries write harmless INFO / WARNING lines to stderr (MediaPipe, Hugging Face). The
    # script stops on errors by default, and with output redirected (> log, | Tee-Object) the first stderr line would then
    # kill the process silently, so these three targets run with errors non-fatal.
    "api" { $ErrorActionPreference = "Continue"; & $Py -m uvicorn backend.main:app --host 0.0.0.0 --port 8020 }
    "workers" { $ErrorActionPreference = "Continue"; $env:PYTHONFAULTHANDLER = "1"; & $Py -m backend.workers.run_all }
    "web-install" { Push-Location frontend; npm install; Pop-Location }
    "web" { $ErrorActionPreference = "Continue"; Push-Location frontend; npm run dev; Pop-Location }
    # The production build is faster and steadier than the dev server (use it for demos): build once, then start it.
    # Stop the dev server first: both use the frontend\.next folder.
    # Demo mode: "own" = only models trained by us, "pretrained" = with the optional Whisper + MediaPipe add-ons. Edits .env; restart the workers.
    "mode-own" { & $Py -m tools.set_mode own }
    "mode-pretrained" { & $Py -m tools.set_mode pretrained }
    "mode" { & $Py -m tools.set_mode }
    "web-build" { $ErrorActionPreference = "Continue"; Push-Location frontend; npm run build; Pop-Location }
    "web-start" { $ErrorActionPreference = "Continue"; Push-Location frontend; npm run start; Pop-Location }
    "start-all" {
        Set-LanIp
        docker compose up -d --wait postgres redis                  # --wait: returns when postgres/redis are healthy
        docker compose up -d --force-recreate livekit caddy         # recreate so they pick up the current address
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$PSScriptRoot'; .\tasks.ps1 api"
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$PSScriptRoot'; .\tasks.ps1 workers"
        Start-Process powershell -ArgumentList "-NoExit", "-Command", "Set-Location '$PSScriptRoot'; .\tasks.ps1 web"
        Write-Output "InterLens: http://localhost:3020 (this laptop)  |  https://$($env:MBAS_LAN_IP):8444 (second laptop)"
    }
    # ---- Stage 10: evaluation + documents
    "reports" { & $Py -m tools.model_report; & $Py -m tools.bias_check; & $Py -m tools.user_study --demo }
    default {
        Write-Output "targets: venv install up down ps questions recorder-api recorder-web-install recorder-web check <dir> test labelstudio labelstudio-import labels datasets splits stats data eda train-all export retrain-live demo-webcam demo-mic api workers web-install web start-all reports"
    }
}
