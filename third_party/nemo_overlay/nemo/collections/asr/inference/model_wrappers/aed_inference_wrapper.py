# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omegaconf import open_dict

from nemo.collections.asr.inference.utils.device_utils import setup_device

if TYPE_CHECKING:
    from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel
    from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis


class AEDInferenceWrapper:
    """
    Inference wrapper for EncDecMultiTaskModel (AED / Canary family).

    Provides a clean encode / decode interface that separates the acoustic
    encoder from the autoregressive text decoder, mirroring the pattern used
    by SALMASRInferenceWrapper but adapted to the NeMo AED architecture.
    """

    def __init__(
        self,
        model_name: str,
        device: str = 'cuda',
        device_id: int = 0,
        compute_dtype: str = 'bfloat16',
        use_amp: bool = True,
        source_lang: str | None = None,
        target_lang: str | None = None,
        task: str | None = None,
        pnc: str | None = None,
    ):
        """
        Initialize the AED ASR inference wrapper.

        Args:
            model_name: Path to a .nemo checkpoint file or a model name on NGC/HuggingFace.
            device: Device to run on ('cuda' or 'cpu').
            device_id: GPU device index when device='cuda'.
            compute_dtype: Dtype string, e.g. 'bfloat16' or 'float32'.
            use_amp: Whether to use torch.amp.autocast during inference.
            source_lang: Default source language for the Canary prompt (e.g. 'en').
            target_lang: Default target language for the Canary prompt (e.g. 'en').
            task: Default task token (e.g. 'asr' or 'ast').
            pnc: Default punctuation-and-capitalisation flag ('yes' or 'no').
        """
        self.device_str, self.device_id, self.compute_dtype = setup_device(device.strip(), device_id, compute_dtype)
        self.use_amp = use_amp
        self.device = torch.device(self.device_str)

        self.aed_model = self.load_model(model_name, self.device)
        self.tokenizer = self.aed_model.tokenizer
        self.set_dither_to_zero()

        # Store prompt defaults; only non-None values override model defaults.
        self._prompt_defaults: dict[str, str] = {}
        for key, val in [('source_lang', source_lang), ('target_lang', target_lang), ('task', task), ('pnc', pnc)]:
            if val is not None:
                self._prompt_defaults[key] = val

        # AMP needs float32 for numerics when use_amp=True (autocast handles the rest).
        self.cast_dtype = torch.float32 if use_amp else self.compute_dtype
        self.aed_model.to(self.cast_dtype)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_model(model_name: str, device: torch.device) -> EncDecMultiTaskModel:
        """
        Load an EncDecMultiTaskModel from a .nemo checkpoint or from pretrained.

        Args:
            model_name: Path ending in '.nemo', or an NGC / HF model name.
            device: Target device.

        Returns:
            Loaded model in eval mode.
        """
        try:
            from nemo.collections.asr.models.aed_multitask_models import EncDecMultiTaskModel

            if model_name.endswith('.nemo'):
                model = EncDecMultiTaskModel.restore_from(model_name, map_location=device)
            else:
                model = EncDecMultiTaskModel.from_pretrained(model_name, map_location=device)
            model.eval()
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load AED model from '{model_name}': {e}") from e

    # ------------------------------------------------------------------
    # Audio / stride helpers
    # ------------------------------------------------------------------

    def get_window_stride(self) -> float:
        """Returns the feature-extraction window stride in seconds."""
        return self.aed_model.cfg.preprocessor.window_stride

    def get_subsampling_factor(self) -> int:
        """Returns the encoder's temporal subsampling factor."""
        return self.aed_model.encoder.subsampling_factor

    def get_model_stride(self, in_secs: bool = False, in_milliseconds: bool = False) -> float:
        """
        Returns the effective per-token duration (window_stride × subsampling_factor).

        Args:
            in_secs: Return value in seconds.
            in_milliseconds: Return value in milliseconds.

        Returns:
            Model stride as a float.
        """
        if in_secs and in_milliseconds:
            raise ValueError("Specify at most one of in_secs or in_milliseconds.")
        stride = self.get_window_stride() * self.get_subsampling_factor()
        if in_secs:
            return stride
        if in_milliseconds:
            return stride * 1000.0
        return stride

    def set_dither_to_zero(self) -> None:
        """Disable dithering in the feature extractor for deterministic inference."""
        self.aed_model.preprocessor.featurizer.dither = 0.0
        with open_dict(self.aed_model.cfg):
            self.aed_model.cfg.preprocessor.dither = 0.0

    # ------------------------------------------------------------------
    # Tokenizer helpers
    # ------------------------------------------------------------------

    @property
    def word_separator(self) -> str:
        """Returns the word separator string (space for BPE models)."""
        return ' '

    @property
    def word_separator_ids(self) -> list[int]:
        """
        Returns the token id(s) corresponding to the word separator.

        The CanaryMultilingualTokenizer requires a lang_id argument to
        text_to_ids. We use the default target_lang when available. If no
        language is configured (or the call fails for any reason), we fall
        back to an empty list — lcs_merge treats None / [] as "no separator",
        which is safe.
        """
        target_lang = self._prompt_defaults.get('target_lang')
        if target_lang is None:
            return []
        try:
            return self.tokenizer.text_to_ids(self.word_separator, target_lang)
        except TypeError:
            # Standard (non-multilingual) tokenizer — no lang_id needed.
            return self.tokenizer.text_to_ids(self.word_separator)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_decoder_input_ids(
        self,
        batch_size: int,
        source_lang: str | None = None,
        target_lang: str | None = None,
        task: str | None = None,
        pnc: str | None = None,
    ) -> torch.Tensor:
        """
        Build the decoder prompt token ids for a batch.

        Uses the model's own default slot values (via ``get_default_dialog_slots``)
        as the base, then overlays instance-level defaults set in ``__init__``,
        then overlays any per-call arguments. This mirrors the slot-filling logic
        in ``EncDecMultiTaskModel._transcribe_forward`` and handles models with
        extra prompt slots (e.g. ``romanized``, ``itn``, ``emotion``).

        Args:
            batch_size: Number of sequences in the batch.
            source_lang: Override source language (e.g. 'hi').
            target_lang: Override target language (e.g. 'en').
            task: Override task ('asr' or 'ast').
            pnc: Override punctuation flag ('yes' or 'no').

        Returns:
            Tensor of shape (batch_size, prompt_length) on self.device.
        """
        # Start from the model's own defaults (covers all required slots).
        turns = self.aed_model.prompt.get_default_dialog_slots()

        # Build the override dict from instance defaults + per-call args.
        overrides: dict[str, str] = dict(self._prompt_defaults)
        for key, val in [('source_lang', source_lang), ('target_lang', target_lang), ('task', task), ('pnc', pnc)]:
            if val is not None:
                overrides[key] = val

        # Merge overrides into each turn's slots (only set non-None overrides).
        for turn in turns:
            turn["slots"] = dict(turn["slots"])  # shallow copy to avoid mutating model state
            for key, val in overrides.items():
                if key in turn["slots"]:
                    turn["slots"][key] = val

        context_ids = (
            self.aed_model.prompt.encode_dialog(turns=turns)["context_ids"]
            .unsqueeze(0)
            .repeat(batch_size, 1)
            .to(self.device)
        )
        return context_ids

    # ------------------------------------------------------------------
    # Encode / decode primitives
    # ------------------------------------------------------------------

    def encode(
        self,
        processed_signal: torch.Tensor,
        processed_signal_length: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run the audio encoder on pre-computed acoustic features.

        Args:
            processed_signal: Feature tensor of shape (B, D, T).
            processed_signal_length: Valid frame counts of shape (B,).

        Returns:
            Tuple of (enc_states, enc_mask, encoded_len):
                enc_states  – shape (B, T', D'), projected encoder output.
                enc_mask    – boolean mask of shape (B, T').
                encoded_len – valid length counts of shape (B,).
        """
        if processed_signal.device != self.device:
            processed_signal = processed_signal.to(self.device)
        if processed_signal_length.device != self.device:
            processed_signal_length = processed_signal_length.to(self.device)

        with (
            torch.amp.autocast(device_type=self.device_str, dtype=self.compute_dtype, enabled=self.use_amp),
            torch.inference_mode(),
            torch.no_grad(),
        ):
            _, encoded_len, enc_states, enc_mask = self.aed_model.forward(
                processed_signal=processed_signal.to(self.cast_dtype),
                processed_signal_length=processed_signal_length,
            )
        return enc_states, enc_mask, encoded_len

    def decode(
        self,
        enc_states: torch.Tensor,
        enc_mask: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        return_hypotheses: bool = False,
    ) -> list[Hypothesis]:
        """
        Autoregressively decode encoder output into text hypotheses.

        Args:
            enc_states: Projected encoder output of shape (B, T', D').
            enc_mask: Encoder mask of shape (B, T').
            decoder_input_ids: Prompt token ids of shape (B, prompt_len).
            return_hypotheses: If True, return full Hypothesis objects.

        Returns:
            List of Hypothesis (one per batch element) when return_hypotheses=True,
            otherwise a list of decoded text strings.
        """
        with torch.inference_mode():
            hypotheses = self.aed_model.decoding.decode_predictions_tensor(
                encoder_hidden_states=enc_states,
                encoder_input_mask=enc_mask,
                decoder_input_ids=decoder_input_ids,
                return_hypotheses=return_hypotheses,
            )
        return hypotheses

    # ------------------------------------------------------------------
    # High-level generate (encode + decode in one call)
    # ------------------------------------------------------------------

    def generate(
        self,
        audios: torch.Tensor,
        audio_lens: torch.Tensor,
        source_lang: str | None = None,
        target_lang: str | None = None,
        task: str | None = None,
        pnc: str | None = None,
        return_hypotheses: bool = True,
    ) -> list[Hypothesis]:
        """
        Encode raw waveforms and decode transcripts in one call.

        Args:
            audios: Raw audio tensor of shape (B, num_samples).
            audio_lens: Valid sample counts of shape (B,).
            source_lang: Override source language.
            target_lang: Override target language.
            task: Override task ('asr' or 'ast').
            pnc: Override punctuation flag ('yes' or 'no').
            return_hypotheses: Return Hypothesis objects (True) or text strings (False).

        Returns:
            List of Hypothesis objects or text strings (one per batch element).
        """
        if audios.device != self.device:
            audios = audios.to(self.device)
        if audio_lens.device != self.device:
            audio_lens = audio_lens.to(self.device)

        with (
            torch.amp.autocast(device_type=self.device_str, dtype=self.compute_dtype, enabled=self.use_amp),
            torch.inference_mode(),
            torch.no_grad(),
        ):
            # Forward pass with raw waveform; the model's preprocessor runs internally.
            _, encoded_len, enc_states, enc_mask = self.aed_model.forward(
                input_signal=audios.to(self.cast_dtype),
                input_signal_length=audio_lens,
            )

        decoder_input_ids = self.build_decoder_input_ids(
            batch_size=audios.shape[0],
            source_lang=source_lang,
            target_lang=target_lang,
            task=task,
            pnc=pnc,
        )

        return self.decode(enc_states, enc_mask, decoder_input_ids, return_hypotheses=return_hypotheses)
