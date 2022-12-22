import json
import base64
import urllib.request

from fastapi import FastAPI, File, Form, HTTPException

app = FastAPI()

@app.post("/image/generate")
async def image_generate(
  pass
