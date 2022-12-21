from typing import Dict, List, Any
import base64
import keras_cv

class EndpointHandler():
    def __init__(self, path=""):
        self.sd = keras_cv.models.StableDiffusion(img_width=512, img_height=512)
    
    def __call__(self, data: Dict[str, Any]) -> str:
        # get inputs 
        prompt = data.pop("inputs", data)
        batch_size = data.pop("batch_size", 1)

        # run normal prediction
        images = self.sd.text_to_image(prompt, batch_size=batch_size)
        return base64.b64encode(images.tobytes()).decode()