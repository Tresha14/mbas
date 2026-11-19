# Run InterLens on a new laptop (Windows)

## What you need

| Thing | Where it comes from |
| --- | --- |
| The code, config, tokenizers, question bank, synthetic dataset | GitHub: https://github.com/Tresha14/mbas |
| The trained models (`checkpoints/`, `ml/export/onnx/`, `data/processed/lm/`) | **not on GitHub** (too big): `mbas_models_2026-11-19.zip` (v2, 121 MB) from Tresha's Google Drive |
| Python **3.11** | https://www.python.org/downloads/ (tick "Add python.exe to PATH") |
| Node.js **20 or newer** | https://nodejs.org (LTS) |
| Docker Desktop | https://www.docker.com/products/docker-desktop |
| Git | https://git-scm.com |
| Google Chrome | for the interviewer and candidate pages |

An NVIDIA GPU is optional: the app runs on the CPU too (slower vision and speech, still live).

## First time (about 20 minutes)

All commands in **PowerShell**, inside the project folder.

1. **Get the code**
   ```powershell
   git clone https://github.com/Tresha14/mbas.git
   cd mbas
   ```
   (Already have it? `git pull` instead.)

2. **Add the trained models.** Extract `mbas_models_2026-11-19.zip` **into the `mbas` folder** (the one with
   `README.md`), merging folders. Afterwards these exist: `checkpoints\asr\averaged.pt`, `ml\export\onnx\`,
   `data\processed\lm\interview_4gram.arpa`. Use the **v2** zip; if an old `checkpoints\asr` folder with
   `epoch_*.pt` files is there, it can be deleted.

3. **Settings file**
   ```powershell
   copy .env.example .env
   ```
   Nothing to edit for one laptop. (`MBAS_LAN_IP` is detected automatically when you start.)

4. **Python environment**
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\python -m pip install --upgrade pip
   ```
   With an NVIDIA GPU:
   ```powershell
   .\.venv\Scripts\python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```
   Without one:
   ```powershell
   .\.venv\Scripts\python -m pip install torch torchaudio
   ```
   Then:
   ```powershell
   .\.venv\Scripts\python -m pip install -r requirements.txt
   ```

5. **Web app packages**
   ```powershell
   cd frontend; npm install; cd ..
   ```

6. **Check everything is in place**
   ```powershell
   .\.venv\Scripts\python -m tools.preflight
   ```
   It must end with **READY**. A `FAIL` line says what is missing (usually the models zip from step 2).

## Every time you run it

1. Open **Docker Desktop** and wait until it says it is running.
2. In the `mbas` folder:
   ```powershell
   .\tasks.ps1 start-all
   ```
   It starts the database, Redis and LiveKit in Docker, then opens three windows (API, workers, web). Keep them
   open. If Windows says running scripts is disabled, use
   `powershell -ExecutionPolicy Bypass -File .\tasks.ps1 start-all` instead.
3. Wait until the **workers** window shows `[speech] ready` and `[vision] ready` (about 30 s the first time), then
   open **http://localhost:3020** in Chrome.
4. Sign in: create an account (**Create an account**), or create the demo account once with
   ```powershell
   .\.venv\Scripts\python -m tools.seed_demo
   ```
   and sign in with `MBAS_DEMO_EMAIL` / `MBAS_DEMO_PASSWORD` from your `.env`.

Before a demo, `.\.venv\Scripts\python -m tools.preflight --services` checks the running services too.

## Running an interview

1. **New session**: title, candidate name, and the **meeting platform** (InterLens room, Google Meet, Zoom, Teams).
2. **InterLens room**: copy the **Invite link** and send it to the candidate. **Meet / Zoom / Teams**: host the call
   there as usual; in the cockpit press **Connect Meet / Zoom / Teams** and follow the three steps shown.
3. Open the session (**Open**), press **Start analysis**, pick questions from the list, press **Answer ended** after
   each answer, then **End interview** and **View report**.

Candidate on a second laptop on the same Wi-Fi: use `https://<this laptop's IP>:8444/join/...` (the start-all
window prints the address) and allow TCP 8444 and UDP 7882 in Windows Firewall on the host laptop.

## If something goes wrong

| Problem | Fix |
| --- | --- |
| A page looks old after `git pull` | Ctrl + Shift + R in Chrome |
| "could not establish pc connection" in the cockpit | Start with `.\tasks.ps1 start-all` (not plain `docker compose up`) while connected to Wi-Fi / LAN |
| "model not trained" / preflight FAIL on a checkpoint | Extract the v2 models zip into the `mbas` folder |
| Port already in use (5442, 6389, 7880, 3020, 8020) | Close the other program, or `docker compose down` and start again |
| Workers window shows an error | Run `.\.venv\Scripts\python -m tools.preflight` and fix what it reports |
