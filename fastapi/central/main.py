import json
import aiohttp

from fastapi import FastAPI

app = FastAPI()

async def fetch(session, url, payload):
	async with session.get(url, params=payload) as response:
			return await response.content

@app.post("/image/generate")
async def image_generate(
	prompt: str = None,
	batch_size: int = None):
  
	if prompt is None:
		print("prompt is not passed")

	if batch_size is None:
		print("batch_size is not passed")

	async with aiohttp.ClientSession() as session:
		# text encoding
		payload = {
			"prompt": prompt,
			"batch_size": batch_size
		}

		content = await fetch(session, 'http://python.org', payload)
		content = json.loads(content)

		# diffusion
		payload = {
			"context": content["context"],
			"u_context": content["unconditional_context"],
			"batch_size": batch_size
		}

		content = await fetch(session, 'http://python.org', payload)
		content = json.loads(content)

		# decoding
		payload = {
			"latent": content["latent"],
			"batch_size": batch_size
		}

		content = await fetch(session, 'http://python.org', payload)
		content = json.loads(content)
		return content["images"]

