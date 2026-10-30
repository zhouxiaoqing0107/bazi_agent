uvicorn main:app --host 127.0.0.1 --port 8000
& "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://127.0.0.1:8000