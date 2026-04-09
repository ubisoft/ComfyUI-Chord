import torch
from pathlib import Path
from torchvision.transforms import v2

from diffusers import UNet2DConditionModel, AutoencoderKL, DDIMScheduler
from transformers import CLIPTextModel, CLIPTextConfig, CLIPTokenizer

from . import register
from .base import Base


def apply_padding(model, mode):
    for layer in [layer for _, layer in model.named_modules() if isinstance(layer, torch.nn.Conv2d)]:
        if mode == 'circular':
            layer.padding_mode = 'circular'
        else:
            layer.padding_mode = 'zeros'
    return model


def freeze(model):
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


@register("stable_diffusion")
class StableDiffusion(Base):
    def setup(self):
        hf_key = self.config.get("hf_key", None)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        fp16 = self.config.get("fp16", True)
        self.dtype = torch.bfloat16 if fp16 else torch.float32
        vae_padding = self.config.get("vae_padding", "zeros")

        self.sd_version = self.config.get("version", 2.1)
        local_files_only = False
        if hf_key is not None:
            print(f"[INFO] using hugging face custom model key: {hf_key}")
            model_key = hf_key
            local_files_only = True
        elif str(self.sd_version) == "2.1":
            # model_key = "stabilityai/stable-diffusion-2-1"
            # StabilityAI deleted the original 2.1 model from HF, use a community version
            model_key = str(Path(__file__).parent.parent.parent / "config")
        else:
            raise ValueError(
                f"Stable-diffusion version {self.sd_version} not supported."
            )

        # Load components separately to avoid download unnecessary weights
        # 1. UNet (diffusion backbone)
        unet_config = UNet2DConditionModel.load_config(model_key, subfolder="unet")
        self.unet = UNet2DConditionModel.from_config(unet_config, local_files_only=local_files_only)
        self.unet.to(self.device, dtype=self.dtype).eval()
        # 2. VAE (image autoencoder)
        vae_config = AutoencoderKL.load_config(model_key, subfolder="vae")
        self.vae = AutoencoderKL.from_config(vae_config, local_files_only=local_files_only)
        self.vae.to(self.device, dtype=self.dtype).eval()
        self.vae = apply_padding(freeze(self.vae), vae_padding)
        # 3. Text encoder (CLIP)
        text_encoder_config = CLIPTextConfig.from_pretrained(model_key, subfolder="text_encoder", local_files_only=local_files_only)
        self.text_encoder = CLIPTextModel(text_encoder_config)
        self.text_encoder = self.text_encoder.to(self.device, dtype=self.dtype).eval()
        # 4. Tokenizer (CLIP tokenizer, this one has vocab so from_pretrained is needed)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_key, subfolder="tokenizer", local_files_only=local_files_only)
        # 5. Scheduler
        scheduler_config = DDIMScheduler.load_config(model_key, subfolder="scheduler")
        scheduler_config["prediction_type"] = "v_prediction"
        scheduler_config["timestep_spacing"] = "trailing"
        scheduler_config["rescale_betas_zero_snr"] = True
        self.scheduler = DDIMScheduler.from_config(scheduler_config)

    def encode_text(self, prompt, padding_mode="do_not_pad"):
        # prompt: [str]
        self.text_encoder = self.text_encoder.to(self.device)  # force materialization for MPS
        inputs = self.tokenizer(
            prompt,
            padding=padding_mode,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings    

    def decode_latents(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    def encode_imgs(self, imgs):
        if imgs.shape[1] == 1: # for grayscale maps
            imgs = v2.functional.grayscale_to_rgb(imgs)
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents
    
    def encode_imgs_deterministic(self, imgs):
        if imgs.shape[1] == 1: # for grayscale maps
            imgs = v2.functional.grayscale_to_rgb(imgs)
        imgs = 2 * imgs - 1
        h = self.vae.encoder(imgs)
        moments = self.vae.quant_conv(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        latents = mean * self.vae.config.scaling_factor
        return latents