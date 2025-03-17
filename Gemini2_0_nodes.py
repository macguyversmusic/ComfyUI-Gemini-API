# import os
# import base64
# import io
# import json
# import torch
# import numpy as np
# from PIL import Image
# import requests
# from io import BytesIO
# from google import genai
# from google.genai import types
# import traceback

# class GeminiImageGenerator:
#     @classmethod
#     def INPUT_TYPES(cls):
#         return {
#             "required": {
#                 "prompt": ("STRING", {"multiline": True}),
#                 "api_key": ("STRING", {"default": "", "multiline": False}),
#                 "model": (["models/gemini-2.0-flash-exp"], {"default": "models/gemini-2.0-flash-exp"}),
#                 "width": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8}),
#                 "height": ("INT", {"default": 1024, "min": 512, "max": 2048, "step": 8}),
#                 "temperature": ("FLOAT", {"default": 1, "min": 0.0, "max": 2.0, "step": 0.05}),
#             },
#             "optional": {
#                 "seed": ("INT", {"default": 66666666, "min": 0, "max": 2147483647}),
#                 "image1": ("IMAGE",),
#                 "image2": ("IMAGE",),
#                 "image3": ("IMAGE",),
#                 "image4": ("IMAGE",),
#             }
#         }


#     RETURN_TYPES = ("IMAGE", "STRING")
#     RETURN_NAMES = ("image", "API Respond")
#     FUNCTION = "generate_image"
#     CATEGORY = "Google-Gemini"
    
#     def __init__(self):
#         """Initialize logging system and API key storage."""
#         self.log_messages = []  # Global log message storage
#         # Get the node's directory
#         self.node_dir = os.path.dirname(os.path.abspath(__file__))
#         self.key_file = os.path.join(self.node_dir, "gemini_api_key.txt")
        
#         # Check the version of google-genai
#         try:
#             import importlib.metadata
#             genai_version = importlib.metadata.version('google-genai')
#             self.log(f"Current google-genai version: {genai_version}")
            
#             # Check PIL/Pillow version
#             try:
#                 import PIL
#                 self.log(f"Current PIL/Pillow version: {PIL.__version__}")
#             except Exception as e:
#                 self.log(f"Unable to check PIL/Pillow version: {str(e)}")
            
#             # Check if the version meets minimum requirements
#             from packaging import version
#             if version.parse(genai_version) < version.parse('1.5.0'):
#                 self.log("Warning: google-genai version is too low; please upgrade to the latest version")
#                 self.log("Suggested command: pip install -q -U google-genai")
            
#             # Check if PIL/Pillow version meets requirements
#             try:
#                 if version.parse(PIL.__version__) < version.parse('9.5.0'):
#                     self.log("Warning: PIL/Pillow version is too low; please upgrade to 9.5.0 or higher")
#                     self.log("Suggested command: pip install -U Pillow>=9.5.0")
#             except Exception:
#                 pass
#         except Exception as e:
#             self.log(f"Unable to check version information: {e}")
    
#     def log(self, message):
#         """Global logging function: record message to the log list."""
#         if hasattr(self, 'log_messages'):
#             self.log_messages.append(message)
#         return message
    
#     def get_api_key(self, user_input_key):
#         """Obtain the API key, prioritizing the user input."""
#         # If the user provides a valid key, use and save it
#         if user_input_key and len(user_input_key) > 10:
#             self.log("Using user provided API key")
#             # Save to file
#             try:
#                 with open(self.key_file, "w") as f:
#                     f.write(user_input_key)
#                 self.log("Saved API key to node directory")
#             except Exception as e:
#                 self.log(f"Failed to save API key: {e}")
#             return user_input_key
            
#         # If no user input, try reading from file
#         if os.path.exists(self.key_file):
#             try:
#                 with open(self.key_file, "r") as f:
#                     saved_key = f.read().strip()
#                 if saved_key and len(saved_key) > 10:
#                     self.log("Using saved API key")
#                     return saved_key
#             except Exception as e:
#                 self.log(f"Failed to read saved API key: {e}")
                
#         # If none available, return empty string
#         self.log("Warning: No valid API key provided")
#         return ""
    
#     def generate_empty_image(self, width, height):
#         """Generate a blank RGB image tensor in ComfyUI-compatible format [B,H,W,C]."""
#         # Create an image tensor in ComfyUI standard format
#         # ComfyUI expects [batch, height, width, channels] format!
#         empty_image = np.ones((height, width, 3), dtype=np.float32) * 0.2
#         tensor = torch.from_numpy(empty_image).unsqueeze(0)  # [1, H, W, 3]
        
#         self.log(f"Created ComfyUI-compatible empty image: shape={tensor.shape}, dtype={tensor.dtype}")
#         return tensor
    
#     def validate_and_fix_tensor(self, tensor, name="Image"):
#         """Validate and fix tensor format to ensure full compatibility with ComfyUI."""
#         try:
#             # Basic shape check
#             if tensor is None:
#                 self.log(f"Warning: {name} is None")
#                 return None
                
#             self.log(f"Validating {name}: shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}")
            
#             # Ensure correct shape: [B, C, H, W]
#             if len(tensor.shape) != 4:
#                 self.log(f"Error: {name} shape is incorrect: {tensor.shape}")
#                 return None
                
#             if tensor.shape[1] != 3:
#                 self.log(f"Error: {name} channel count is not 3: {tensor.shape[1]}")
#                 return None
                
#             # Ensure type is float32
#             if tensor.dtype != torch.float32:
#                 self.log(f"Fixing {name} type: {tensor.dtype} -> torch.float32")
#                 tensor = tensor.to(dtype=torch.float32)
                
#             # Ensure memory is contiguous
#             if not tensor.is_contiguous():
#                 self.log(f"Fixing {name} memory layout: making it contiguous")
#                 tensor = tensor.contiguous()
                
#             # Ensure value range is between 0 and 1
#             min_val = tensor.min().item()
#             max_val = tensor.max().item()
            
#             if min_val < 0 or max_val > 1:
#                 self.log(f"Fixing {name} value range: [{min_val}, {max_val}] -> [0, 1]")
#                 tensor = torch.clamp(tensor, 0.0, 1.0)
                
#             return tensor
#         except Exception as e:
#             self.log(f"Error validating tensor: {e}")
#             traceback.print_exc()
#             return None
    
#     def generate_image(self, prompt, api_key, model, width, height, temperature, seed=66666666, image1=None, image2=None, image3=None, image4=None):
#         """Generate an image using simplified API key management."""
#         response_text = ""
        
#         # Reset log messages
#         self.log_messages = []
        
#         try:
#             # Obtain API key
#             actual_api_key = self.get_api_key(api_key)
            
#             if not actual_api_key:
#                 error_message = "Error: No valid API key provided. Please enter your Google API key in the node or ensure one is saved."
#                 self.log(error_message)
#                 full_text = "## Error\n" + error_message + "\n\n## Instructions\n1. Enter your Google API key in the node\n2. The key will be automatically saved in the node directory for future use."
#                 return (self.generate_empty_image(width, height), full_text)
            
#             # Create client instance
#             client = genai.Client(api_key=actual_api_key)
            
#             # Process seed value
#             if seed == 0:
#                 import random
#                 seed = random.randint(1, 2**31 - 1)
#                 self.log(f"Generated random seed value: {seed}")
#             else:
#                 self.log(f"Using specified seed value: {seed}")
            
#             # Build a prompt with size information
#             aspect_ratio = width / height
#             if aspect_ratio > 1:
#                 orientation = "landscape (horizontal)"
#             elif aspect_ratio < 1:
#                 orientation = "portrait (vertical)"
#             else:
#                 orientation = "square"

#             simple_prompt = f"Create a detailed image of: {prompt}. Generate the image in {orientation} orientation with exact dimensions of {width}x{height} pixels. Ensure the composition fits properly within these dimensions without stretching or distortion."
            
#             # Configure generation parameters using the specified temperature
#             gen_config = types.GenerateContentConfig(
#                 temperature=temperature,
#                 seed=seed,
#                 response_modalities=['Text', 'Image']
#             )
            
#             # Log temperature and seed settings
#             self.log(f"Using temperature value: {temperature}, seed: {seed}")
            
#             # Process reference image
#             contents = [{"text": simple_prompt}]  # start with text prompt
#             image_inputs = [image1, image2, image3, image4]
#             has_reference = False
            
#             for idx, img in enumerate(image_inputs, start=1):
#                 if img is not None:
#                     try:
#                         if len(img.shape) == 4 and img.shape[0] == 1:  # Check [1,H,W,3] format
#                             input_image = img[0].cpu().numpy()
#                             input_image = (input_image * 255).astype(np.uint8)
#                             pil_image = Image.fromarray(input_image)
            
#                             img_byte_arr = BytesIO()
#                             pil_image.save(img_byte_arr, format='PNG')
#                             img_byte_arr.seek(0)
#                             image_bytes = img_byte_arr.read()
            
#                             img_part = {"inline_data": {"mime_type": "image/png", "data": image_bytes}}
#                             contents.append(img_part)
#                             self.log(f"Reference image {idx} processed and added.")
#                             has_reference = True
#                         else:
#                             self.log(f"Reference image {idx} format incorrect: {img.shape}")
#                     except Exception as img_error:
#                         self.log(f"Reference image {idx} processing error: {str(img_error)}")
            
#             # Final API call remains the same
#             response = client.models.generate_content(
#                 model="models/gemini-2.0-flash-exp",
#                 contents=contents,
#                 config=gen_config
#             )

#             # Process API response
#             self.log("API response received, processing...")
            
#             if not hasattr(response, 'candidates') or not response.candidates:
#                 self.log("No candidates found in API response")
#                 full_text = "\n".join(self.log_messages) + "\n\nAPI returned an empty response"
#                 return (self.generate_empty_image(width, height), full_text)
            
#             # Check for image data in the response
#             image_found = False
            
#             # Iterate over response parts
#             for part in response.candidates[0].content.parts:
#                 # Check if it's a text part
#                 if hasattr(part, 'text') and part.text is not None:
#                     text_content = part.text
#                     response_text += text_content
#                     self.log(f"API returned text: {text_content[:100]}..." if len(text_content) > 100 else text_content)
                
#                 # Check if it's an image part
#                 elif hasattr(part, 'inline_data') and part.inline_data is not None:
#                     self.log("Processing API image data")
#                     try:
#                         # Get image data
#                         image_data = part.inline_data.data
#                         mime_type = part.inline_data.mime_type if hasattr(part.inline_data, 'mime_type') else "Unknown"
#                         self.log(f"Image data type: {type(image_data)}, MIME type: {mime_type}, data length: {len(image_data) if image_data else 0}")
                        
#                         # Log the first 8 bytes for diagnosis
#                         if image_data and len(image_data) > 8:
#                             hex_prefix = ' '.join([f'{b:02x}' for b in image_data[:8]])
#                             self.log(f"First 8 bytes of image data: {hex_prefix}")
                            
#                             # Check for Base64-encoded PNG data (example signature check)
#                             if hex_prefix.startswith('69 56 42 4f 52'):
#                                 try:
#                                     self.log("Detected Base64-encoded PNG, decoding...")
#                                     base64_str = image_data.decode('utf-8', errors='ignore')
#                                     image_data = base64.b64decode(base64_str)
#                                     self.log(f"Base64 decoding successful, new data length: {len(image_data)}")
#                                 except Exception as e:
#                                     self.log(f"Base64 decoding failed: {str(e)}")
                        
#                         try:
#                             # Initialize BytesIO with image data
#                             buffer = BytesIO(image_data)
                            
#                             # Attempt to open the image
#                             pil_image = Image.open(buffer)
#                             self.log(f"Image opened successfully: {pil_image.width}x{pil_image.height}, format: {pil_image.format}")
                            
#                             # Ensure the image is in RGB mode
#                             if pil_image.mode != 'RGB':
#                                 pil_image = pil_image.convert('RGB')
                            
#                             # Resize if needed
#                             if pil_image.width != width or pil_image.height != height:
#                                 pil_image = pil_image.resize((width, height), Image.Resampling.LANCZOS)
                            
#                             # Convert to numpy array and then to a tensor
#                             img_array = np.array(pil_image).astype(np.float32) / 255.0
#                             img_tensor = torch.from_numpy(img_array).unsqueeze(0)
                            
#                             self.log(f"Image successfully converted to tensor, shape: {img_tensor.shape}")
                            
#                             full_text = "## Log\n" + "\n".join(self.log_messages) + "\n\n## API Response\n" + response_text
#                             return (img_tensor, full_text)
                        
#                         except Exception as e:
#                             self.log(f"Failed to open image with BytesIO: {str(e)}")
#                             self.log("Using default empty image due to image processing failure")
#                             img_tensor = self.generate_empty_image(width, height)
#                     except Exception as e:
#                         self.log(f"Image processing error: {e}")
#                         traceback.print_exc()
#             if not image_found:
#                 self.log("No image data found in API response, returning text only")
#                 if not response_text:
#                     response_text = "API returned neither image nor text"
            
#             full_text = "## Log\n" + "\n".join(self.log_messages) + "\n\n## API Response\n" + response_text
#             return (self.generate_empty_image(width, height), full_text)
        
#         except Exception as e:
#             error_message = f"Error during processing: {str(e)}"
#             self.log(f"Gemini image generation error: {str(e)}")
#             full_text = "## Log\n" + "\n".join(self.log_messages) + "\n\n## Error\n" + error_message
#             return (self.generate_empty_image(width, height), full_text)

# # Node registration
# NODE_CLASS_MAPPINGS = {
#     "Google-Gemini": GeminiImageGenerator
# }

# NODE_DISPLAY_NAME_MAPPINGS = {
#     "Google-Gemini": "Gemini 2.0 image"
# }


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

                    img_part = {"inline_data": {"mime_type": "image/png", "data": image_bytes}}
                    contents.append(img_part)
                    has_images = True
                    self.log(f"Image{i} added.")
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

