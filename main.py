import torch
import scipy
from diffusers import AudioLDM2Pipeline


class LDM:
    def __init__(self, prompt: str, negative_prompt: str, seed: int, num_inference_steps: int, audio_length: float,
                 num_waveforms: int):
        # Initialise instance variables for constructor arguments
        self.prompt = prompt # These aren't needed in this function but im keeping them as instance variables for
        # possible future extension requirements
        self.negative_prompt = negative_prompt

        self.seed = seed

        self.num_inference_steps = num_inference_steps
        self.audio_length = audio_length
        self.num_waveforms_on_generate = num_waveforms

        self.repository_id = "cvssp/audioldm2"

        self.pipeline = AudioLDM2Pipeline.from_pretrained(self.repository_id,
                                                          torch_dtype=torch.float16)  # Import BnB and quantize
        self.pipeline = self.pipeline.to("cuda")

        self.generator = torch.Generator("cuda").manual_seed(self.seed)

    def set_seed(self, seed: int):
        self.seed = seed

    def set_num_inference_steps(self, num_inference_steps: int):
        self.num_inference_steps = num_inference_steps

    def set_audio_length(self, audio_length: float):
        self.audio_length = audio_length

    def set_num_waveforms(self, num_waveforms: int):
        self.num_waveforms_on_generate = num_waveforms

    def generate(self, prompt: str, negative_prompt: str, filename: str, rate: int):
        self.prompt = prompt
        self.negative_prompt = negative_prompt

        audio = self.pipeline(prompt, negative_prompt,
                              num_inference_steps = self.num_inference_steps,
                              audio_length_in_s = self.audio_length,
                              num_waveforms_per_prompt = self.num_waveforms_on_generate,
                              generator = self.generator,).audios

        scipy.io.wavefile.write(filename, rate=rate, data=audio[0])


if __name__ == "__main__":
    prompt_inp = "The sound of a dragon breathing fire"
    negative_prompt_inp = "Low Quality"
    seed_inp = 42

    num_inference_inp = 200
    audio_length_inp = 10.0
    num_waveforms_inp = 1

    audio_model = LDM(prompt_inp, negative_prompt_inp, seed_inp, num_inference_inp, audio_length_inp, num_waveforms_inp)

    filename_inp = "ldm.wav"
    rate_inp = 16000

    audio_model.generate(prompt_inp, negative_prompt_inp, filename_inp, rate_inp)
