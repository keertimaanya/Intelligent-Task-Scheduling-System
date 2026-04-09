# Minimal FastAPI app for OpenEnv validation
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"status": "ok"}
