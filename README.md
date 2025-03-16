ComfyUI Gemini API

中文 | English

Used to call the Google Gemini API within ComfyUI.
Installation Instructions
Method 1: Manual Installation

    Clone this repository into ComfyUI's custom_nodes directory:

cd ComfyUI/custom_nodes
git clone https://github.com/CY-CHENYUE/ComfyUI-Gemini-API

Install the required dependencies:

If you are using the ComfyUI portable version

..\..\..\python_embeded\python.exe -m pip install -r requirements.txt

If you are using your own Python environment

    path\to\your\python.exe -m pip install -r requirements.txt

Method 2: Installation via ComfyUI Manager

    Install and open ComfyUI Manager within ComfyUI.
    In the Manager, search for "Gemini API".
    Click the install button.

After installation, restart ComfyUI.
Node Description
Gemini 2.0 image

alt text

A node that generates images through the Gemini API.

Input Parameters:

    prompt (required): The text prompt describing the image you want to generate.
    api_key (required): Your Google Gemini API key (once set, it will be automatically saved).
    model: Model selection.
    width: The width of the generated image (512-2048 pixels).
    height: The height of the generated image (512-2048 pixels).
    temperature: A parameter that controls the diversity of the generated output (0.0-2.0).
    seed (optional): Random seed; specifying a value allows for reproducible results.
    image (optional): Reference image input, used for style guidance.

Output:

    image: The generated image, which can be connected to other nodes in ComfyUI.
    API Respond: Contains processing logs and the text information returned by the API.

Usage Scenarios:

    Creating unique conceptual art.
    Generating images based on text descriptions.
    Creating new images with a consistent style using a reference image.
    Image-based editing operations.

Obtaining an API Key

    Visit Google AI Studio
    Create an account or log in.
    In the "API Keys" section, create a new API key.
    Copy the API key and paste it into the node's api_key parameter (only needs to be entered the first time; it will be automatically saved thereafter).

Temperature Parameter Explanation

    Temperature range: 0.0 to 2.0
    Lower temperatures (closer to 0): Generate more deterministic, predictable results.
    Higher temperatures (closer to 2): Generate more diverse, creative results.
    Default value 1.0: Balances determinism and creativity.

Important Notes

    The API may have usage limits or fees; please refer to Google's official documentation.
    The quality and speed of image generation depend on the status of Google's servers and your network connection.
    The reference image function will provide your image to Google's service, so please be aware of any privacy implications.
    On first use, you need to enter the API key; thereafter it will be automatically stored in the node directory in the file gemini_api_key.txt.

Contact Me

    X (Twitter): @cychenyue
    TikTok: @cychenyue
    YouTube: @CY-CHENYUE
    BiliBili: @CY-CHENYUE
    Xiaohongshu: @CY-CHENYUE
