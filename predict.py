# An example of how to convert a given API workflow into its own Replicate model
# Replace predict.py with this file when building your own workflow

import os
import mimetypes
import json
import shutil
from typing import List
from cog import BasePredictor, Input, Path
from comfyui import ComfyUI
from cog_model_helpers import optimise_images
from cog_model_helpers import seed as seed_helper

OUTPUT_DIR = "/tmp/outputs"
INPUT_DIR = "/tmp/inputs"
COMFYUI_TEMP_OUTPUT_DIR = "ComfyUI/temp"
ALL_DIRECTORIES = [OUTPUT_DIR, INPUT_DIR, COMFYUI_TEMP_OUTPUT_DIR]

mimetypes.add_type("image/webp", ".webp")

# Save your example JSON to the same directory as predict.py
api_json_file = "workflow_api.json"

# Force HF offline
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


class Predictor(BasePredictor):
    def setup(self):
        self.comfyUI = ComfyUI("127.0.0.1:8188")
        self.comfyUI.start_server(OUTPUT_DIR, INPUT_DIR)

        self.comfyUI.handle_weights(
            {},
            weights_to_download=[
                "checkpoints/chatglm3-fp16.safetensors",
                "Kolors",
                "sdxl_vae.safetensors",
            ],
        )

    def filename_with_extension(self, input_file, prefix):
        extension = os.path.splitext(input_file.name)[1]
        return f"{prefix}{extension}"

    def handle_input_file(
        self,
        input_file: Path,
        filename: str = "image.png",
    ):
        shutil.copy(input_file, os.path.join(INPUT_DIR, filename))

    # Update nodes in the JSON workflow to modify your workflow based on the given inputs
    def update_workflow(self, workflow, **kwargs):
        prompts = workflow["3"]["inputs"]
        prompts["prompt"] = kwargs["prompt"]
        prompts["negative_prompt"] = f"nsfw, {kwargs['negative_prompt']}"
        prompts["num_images_per_prompt"] = kwargs["number_of_images"]

        sampler = workflow["2"]["inputs"]
        sampler["seed"] = kwargs["seed"]
        sampler["width"] = kwargs["width"] - (kwargs["width"] % 8)
        sampler["height"] = kwargs["height"] - (kwargs["height"] % 8)
        sampler["steps"] = kwargs["steps"]
        sampler["cfg"] = kwargs["cfg"]
        sampler["scheduler"] = kwargs["scheduler"]

    def predict(
        self,
        prompt: str = Input(
            default="",
        ),
        negative_prompt: str = Input(
            description="Things you do not want to see in your image",
            default="",
        ),
        number_of_images: int = Input(
            description="Number of images to generate",
            default=1,
            ge=1,
            le=10,
        ),
        width: int = Input(
            description="Width of the image",
            default=1024,
            ge=512,
            le=2048,
        ),
        height: int = Input(
            description="Height of the image",
            default=1024,
            ge=512,
            le=2048,
        ),
        steps: int = Input(
            description="Number of inference steps",
            default=25,
            ge=1,
            le=50,
        ),
        cfg: float = Input(
            description="Guidance scale",
            default=5,
            ge=0,
            le=20,
        ),
        scheduler: str = Input(
            description="Scheduler",
            default="EulerDiscreteScheduler",
            choices=[
                "EulerDiscreteScheduler",
                "EulerAncestralDiscreteScheduler",
                "DPMSolverMultistepScheduler",
                "DPMSolverMultistepScheduler_SDE_karras",
                "UniPCMultistepScheduler",
                "DEISMultistepScheduler",
            ],
        ),
        output_format: str = optimise_images.predict_output_format(),
        output_quality: int = optimise_images.predict_output_quality(),
        seed: int = seed_helper.predict_seed(),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        self.comfyUI.cleanup(ALL_DIRECTORIES)
        seed = seed_helper.generate(seed)

        with open(api_json_file, "r") as file:
            workflow = json.loads(file.read())

        self.update_workflow(
            workflow,
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            number_of_images=number_of_images,
            width=width,
            height=height,
            steps=steps,
            cfg=cfg,
            scheduler=scheduler,
        )

        self.comfyUI.connect()
        self.comfyUI.run_workflow(workflow)

        return optimise_images.optimise_image_files(
            output_format, output_quality, self.comfyUI.get_files(OUTPUT_DIR)
        )
