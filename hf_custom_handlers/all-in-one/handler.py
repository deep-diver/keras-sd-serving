from typing import Dict, Any
import sys
import base64
import logging
import keras_cv

class EndpointHandler():
    def __init__(self, path="", version="2"):
        self.sd = self._instantiate_stable_diffusion(version)

        if isinstance(self.sd, str):
            sys.exit(self.sd)
        else:
            self.sd.text_to_image("test prompt", batch_size=1)
            logging.warning(f"Stable Diffusion v{version} is fully loaded")

    def _instantiate_stable_diffusion(self, version: str):
        if version is "1.4":
            return keras_cv.models.StableDiffusion(img_width=512, img_height=512)
        elif version is "2":
            return keras_cv.models.StableDiffusionV2(img_width=512, img_height=512)
        else:
            return f"v{version} is not supported"
    
    def __call__(self, data: Dict[str, Any]) -> str:
        prompt = data.pop("inputs", data)
        batch_size = data.pop("batch_size", 1)

        images = self.sd.text_to_image(prompt, batch_size=batch_size)
        return base64.b64encode(images.tobytes()).decode()