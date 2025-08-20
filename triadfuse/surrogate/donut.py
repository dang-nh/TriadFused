"""
Donut surrogate model adapter

Donut is an OCR-free document understanding transformer that directly
processes document images to generate text outputs.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
from transformers import DonutProcessor, VisionEncoderDecoderModel

from .base import SurrogateModel


class DonutSurrogate(SurrogateModel):
    """
    Donut model adapter for document VQA and information extraction

    Uses the Donut vision-encoder-decoder architecture for OCR-free
    document understanding tasks.
    """

    def __init__(
        self,
        model_name: str = "naver-clova-ix/donut-base",
        device: str | None = None,
        max_length: int = 64,
        use_8bit: bool = False,
    ):
        """
        Initialize Donut surrogate

        Args:
            model_name: HuggingFace model identifier
            device: Device to run on (cuda/cpu)
            max_length: Maximum generation length
            use_8bit: Whether to use 8-bit quantization (requires bitsandbytes)
        """
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length

        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load processor and model
        self.processor = DonutProcessor.from_pretrained(model_name)

        # Load model with error handling
        try:
            if use_8bit and self.device.type == "cuda":
                try:
                    from transformers import BitsAndBytesConfig

                    quantization_config = BitsAndBytesConfig(
                        load_in_8bit=True,
                        bnb_8bit_compute_dtype=torch.float16,
                        bnb_8bit_use_double_quant=True,
                    )
                    self.model = VisionEncoderDecoderModel.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        device_map="auto",
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                    )
                    print("✓ Loaded model with 8-bit quantization")
                except ImportError:
                    print("⚠ bitsandbytes not available, loading in full precision")
                    self.model = VisionEncoderDecoderModel.from_pretrained(
                        model_name, 
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True
                    )
                    self.model.to(self.device)
            else:
                # Load without quantization
                dtype = torch.float16 if self.device.type == "cuda" else torch.float32
                self.model = VisionEncoderDecoderModel.from_pretrained(
                    model_name, 
                    torch_dtype=dtype,
                    low_cpu_mem_usage=True
                )
                self.model.to(self.device)
        except Exception as e:
            print(f"⚠ Failed to load model on {self.device}, falling back to CPU: {e}")
            self.device = torch.device("cpu")
            self.model = VisionEncoderDecoderModel.from_pretrained(
                model_name, 
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            self.model.to(self.device)

        # Set model to training mode for gradient computation
        self.model.train()

        # Cache decoder start token
        self.decoder_start_token_id = self.model.config.decoder_start_token_id
        
        # Get model dtype for input consistency
        self.model_dtype = next(self.model.parameters()).dtype

    def forward_task_loss(
        self, image: torch.Tensor, prompt: str, target: str
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute cross-entropy loss for target generation

        Args:
            image: Input image tensor (B, C, H, W) in range [0, 1]
            prompt: Task prompt (e.g., "<s_docvqa><s_question>What is the total?")
            target: Target answer string

        Returns:
            Loss tensor and auxiliary outputs
        """
        batch_size = image.shape[0]

        # Convert tensor to PIL-like format for processor
        # Processor expects uint8 images
        images_np = (image * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        images_list = [images_np[i] for i in range(batch_size)]

        # Process images
        pixel_values = self.processor(
            images_list, return_tensors="pt"
        ).pixel_values.to(self.device)
        
        # Ensure correct dtype for the model
        pixel_values = pixel_values.to(dtype=self.model_dtype)

        # Prepare decoder inputs and labels
        # Combine prompt and target for full sequence
        full_text = prompt + target

        # Tokenize the full sequence for labels
        labels = self.processor.tokenizer(
            full_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        # For decoder input, use prompt only (model will learn to generate target)
        decoder_input_ids = self.processor.tokenizer(
            prompt,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        # Expand to batch size if needed
        if batch_size > 1:
            decoder_input_ids = decoder_input_ids.expand(batch_size, -1)
            labels = labels.expand(batch_size, -1)

        # Forward pass with labels for loss computation
        outputs = self.model(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
            return_dict=True,
        )

        loss = outputs.loss

        # Collect auxiliary outputs
        aux_outputs = {
            "logits": outputs.logits if hasattr(outputs, "logits") else None,
            "encoder_last_hidden_state": outputs.encoder_last_hidden_state
            if hasattr(outputs, "encoder_last_hidden_state")
            else None,
        }

        return loss, aux_outputs

    @torch.no_grad()
    def predict(self, image: torch.Tensor, prompt: str) -> str:
        """
        Generate text prediction for input image

        Args:
            image: Input image tensor (B, C, H, W)
            prompt: Task prompt

        Returns:
            Generated text response
        """
        # Ensure model is in eval mode for prediction
        self.model.eval()

        # Convert tensor to processor format
        if image.dim() == 3:
            image = image.unsqueeze(0)

        images_np = (image * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        images_list = [images_np[0]]  # Take first image if batch

        # Process image
        pixel_values = self.processor(
            images_list, return_tensors="pt"
        ).pixel_values.to(self.device)
        
        # Ensure correct dtype for the model
        pixel_values = pixel_values.to(dtype=self.model_dtype)

        # Prepare decoder prompt
        decoder_input_ids = self.processor.tokenizer(
            prompt,
            add_special_tokens=True,
            return_tensors="pt",
        ).input_ids.to(self.device)

        # Generate response
        generated_ids = self.model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=self.max_length,
            early_stopping=True,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,  # Greedy decoding for speed
            do_sample=False,
        )

        # Decode generated tokens
        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # Remove prompt from generated text if present
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :].strip()

        # Switch back to train mode
        self.model.train()

        return generated_text

    def tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text using Donut tokenizer"""
        return self.processor.tokenizer(
            text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        ).input_ids.to(self.device)
