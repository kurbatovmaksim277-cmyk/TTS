
Remove-Item -Recurse -Force .venv - удаляеет
python -m venv .venv - создает 
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass -активация venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements_client.txt

python -m pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124 --index-url https://download.pytorch.org/whl/cu124

python -m pip install -r requirements_local.txt

python tts_simple_local.py



