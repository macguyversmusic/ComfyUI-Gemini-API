import os
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from google import genai
from google.genai import types
import traceback

class GeminiImageGenerator:
    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "model": (["models/gemini-2.0-flash-exp"], {"default": "models/gemini-2.0-flash-exp"}),
                "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8}),
                "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8}),
                "temperature": ("FLOAT", {"default": 1, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "seed": ("INT", {"default": 66666666, "min": 0, "max": 2147483647}),
                **{f"image{i}": ("IMAGE",) for i in range(1, 11)},
            }
        }
        return inputs

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "API Respond")
    FUNCTION = "generate_image"
    CATEGORY = "Google-Gemini"

    def __init__(self):
        self.log_messages = []
        self.node_dir = os.path.dirname(os.path.abspath(__file__))
        self.key_file = os.path.join(self.node_dir, "gemini_api_key.txt")

    def log(self, message):
        self.log_messages.append(message)

    def get_api_key(self, user_input_key):
        if user_input_key and len(user_input_key) > 10:
            with open(self.key_file, "w") as f:
                f.write(user_input_key)
            self.log("API key saved.")
            return user_input_key
        if os.path.exists(self.key_file):
            with open(self.key_file, "r") as f:
                return f.read().strip()
        self.log("No valid API key provided.")
        return ""

    def generate_empty_image(self, width, height):
        empty_image = np.ones((height, width, 3), dtype=np.float32) * 0.2
        return torch.from_numpy(empty_image).unsqueeze(0)

    def generate_image(self, prompt, api_key, model, width, height, temperature, seed=66666666, **kwargs):
        self.log_messages = []
        actual_api_key = self.get_api_key(api_key)

        if not actual_api_key:
            error_msg = "No valid API key."
            self.log(error_msg)
            return (self.generate_empty_image(width, height), error_msg)

        client = genai.Client(api_key=actual_api_key)

        if seed == 0:
            import random
            seed = random.randint(1, 2**31 - 1)

        aspect_ratio = width / height
        orientation = "landscape" if aspect_ratio > 1 else "portrait" if aspect_ratio < 1 else "square"
        simple_prompt = (f"Create a detailed image of: {prompt}. Generate in {orientation} orientation, "
                         f"dimensions: {width}x{height} pixels.")
        
        ordinals = ["first", "second", "third", "fourth", "fifth",
                    "sixth", "seventh", "eighth", "ninth", "tenth"]
        
        contents = [{"text": simple_prompt}]
        has_images = False
        
        for i in range(1, 11):
            img_tensor = kwargs.get(f'image{i}')
            if img_tensor is not None and len(img_tensor.shape) == 4:
                try:
                    img_array = (img_tensor[0].cpu().numpy() * 255).astype(np.uint8)
                    pil_image = Image.fromarray(img_array)
                    buffer = BytesIO()
                    pil_image.save(buffer, format='PNG')
                    image_bytes = buffer.getvalue()
        
                    img_part = {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_bytes
                        }
                    }
        
                    # Append image with numeric schema
                    contents.append({"text": f"Image {i}:"})
                    contents.append(img_part)
        
                    # Optionally, add ordinal text schema reference:
                    contents.append({"text": f"This is the {ordinals[i-1]} image."})
        
                    has_images = True
                    self.log(f"Image{i} ({ordinals[i-1]}) added.")
                except Exception as e:
                    self.log(f"Failed to process image{i}: {e}")

        gen_config = types.GenerateContentConfig(
            temperature=temperature,
            seed=seed,
            response_modalities=['Text', 'Image']
        )

        try:
            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=gen_config
            )
            if not response.candidates:
                raise ValueError("API returned no candidates.")

            for part in response.candidates[0].content.parts:
                if hasattr(part, 'inline_data') and part.inline_data:
                    image_data = part.inline_data.data
                    pil_image = Image.open(BytesIO(image_data)).convert('RGB').resize((width, height), Image.Resampling.LANCZOS)
                    img_tensor = torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)
                    return img_tensor, "Success"

            raise ValueError("No image found in API response.")

        except Exception as e:
            self.log(f"API error: {e}")
            traceback.print_exc()
            return self.generate_empty_image(width, height), f"API error: {e}"

NODE_CLASS_MAPPINGS = {"Google-Gemini": GeminiImageGenerator}
NODE_DISPLAY_NAME_MAPPINGS = {"Google-Gemini": "Gemini 2.0 image"}

