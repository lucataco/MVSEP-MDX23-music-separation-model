# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

from cog import BasePredictor, Input, Path
import os
import time
import subprocess
from typing import List

MODEL_CACHE = "models"
MODEL_URL = "https://weights.replicate.delivery/default/zfturbo/mvsep-mdx23.tar"
CHECKPOINTS_CACHE = "/root/.cache/torch/hub/checkpoints/"
CHECKPOINTS_URL = "https://weights.replicate.delivery/default/zfturbo/mvsep-mdx23-checkpoints.tar"

def download_weights(url, dest):
    start = time.time()
    print("downloading url: ", url)
    print("downloading to: ", dest)
    subprocess.check_call(["pget", "-x", url, dest], close_fds=False)
    print("downloading took: ", time.time() - start)

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Download models
        if not os.path.exists(MODEL_CACHE):
            download_weights(MODEL_URL, MODEL_CACHE)
        if not os.path.exists(CHECKPOINTS_CACHE):
            download_weights(CHECKPOINTS_URL, CHECKPOINTS_CACHE)

    def predict(
        self,
        audio: Path = Input(description="Input Audio File"),
    ) -> List[Path]:
        """Run a single prediction on the model"""
        # Clear past runs
        output_folder = "/tmp/results/"
        # Remove output folder if it exists
        if os.path.exists(output_folder):
            os.system("rm -rf " + output_folder)
        # Run MVSEP subprocess
        subprocess.run(["python", "inference.py", "--input_audio", str(audio), "--output_folder", output_folder], check=True)
        # Get list of files in the output folder
        files = os.listdir(output_folder)
        # Return list of files
        output_files = [Path(os.path.join(output_folder, file)) for file in files]
        return output_files
