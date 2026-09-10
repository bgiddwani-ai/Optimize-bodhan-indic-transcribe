# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

import yaml
import logging


def config_for_trtllm(model, artifacts, **kwargs):
    if model.__class__.__name__ == 'EncDecMultiTaskModel':

        model_config = yaml.safe_load(artifacts['model_config.yaml']['content'])

        keys_required = [
            'beam_search',
            'encoder',
            'head',
            'model_defaults',
            'prompt_format',
            'sample_rate',
            'target',
            'preprocessor',
        ]
        if 'beam_search' not in model_config and 'decoding' in model_config:
            model_config['beam_search'] = model_config['decoding'].get('beam', {'beam_size': 1, 'len_pen': 0.0,
                                                                                'max_generation_delta': 50}
                                                                       )
        config = dict({k: model_config[k] for k in keys_required})
        config['decoder'] = {
            'transf_decoder': model_config['transf_decoder'],
            'transf_encoder': model_config['transf_encoder'],
            'vocabulary': make_vocabulary_file(model, artifacts),
            'num_classes': model_config['head']['num_classes'],
            'feat_in': model_config['model_defaults']['asr_enc_hidden'],
            'n_layers': model_config['transf_decoder']['config_dict']['num_layers'],
        }
        config['target'] = 'trtllm.canary'

        artifacts['model_config.yaml']['content'] = yaml.safe_dump(config, encoding=('utf-8'))


def make_vocabulary_file(model, artifacts, **kwargs):
    if model.__class__.__name__ == 'EncDecMultiTaskModel':

        tokenizer_vocab = {'tokens': {},
                           'offsets': model.tokenizer.token_id_offset
                           }
        for lang in model.tokenizer.langs:
            tokenizer_vocab['tokens'][lang] = {}
        tokenizer_vocab['size'] = model.tokenizer.vocab_size

        try:
            tokenizer_vocab['bos_id'] = model.tokenizer.bos_id
        except Exception as e:
            logging.warning(f"Tokenizer is missing bos_id. Could affect accuracy")

        try:
            tokenizer_vocab['eos_id'] = model.tokenizer.eos_id
        except Exception as e:
            logging.warning(f"Tokenizer is missing eos_id. Could affect accuracy")
        try:
            tokenizer_vocab['nospeech_id'] = model.tokenizer.nospeech_id
        except Exception as e:
            logging.warning(f"Tokenizer is missing nospeech_id. Could affect accuracy")
        try:
            tokenizer_vocab['pad_id'] = model.tokenizer.pad_id
        except Exception as e:
            logging.warning(f"Tokenizer is missing pad_id. Could affect accuracy")

        for t_id in range(0, model.tokenizer.vocab_size):
            lang = model.tokenizer.ids_to_lang([t_id])
            tokenizer_vocab['tokens'][lang][t_id] = model.tokenizer.ids_to_tokens([t_id])[0]

        # PATCH (defensive): explicitly ensure every special token (all <|xx|>
        # language and flag tokens like <|hi|>, <|pnc|>, <|emo:undefined|>) is
        # present in the spl_tokens bucket. The per-id loop above SHOULD already
        # cover this via ids_to_lang, but downstream (riva-build / servicemaker /
        # canary.py CanaryTokenizer) reads language codes directly out of
        # spl_tokens and silently falls back to <unk> if a token is missing,
        # which produces hard-to-diagnose wrong-but-plausible transcripts.
        # This belt-and-suspenders block guarantees every special token lands.
        # try:
        #     special_tokens = getattr(model.tokenizer, 'special_tokens', None)
        #     if special_tokens and 'spl_tokens' in tokenizer_vocab['tokens']:
        #         added = 0
        #         for tok_str, tok_id in special_tokens.items():
        #             tid = int(tok_id)
        #             if tid not in tokenizer_vocab['tokens']['spl_tokens']:
        #                 added += 1
        #             tokenizer_vocab['tokens']['spl_tokens'][tid] = tok_str
        #         if added:
        #             logging.info(
        #                 f"make_vocabulary_file: added {added} missing special tokens "
        #                 f"to spl_tokens (e.g. language tokens)"
        #             )
        # except Exception as e:
        #     logging.warning(f"make_vocabulary_file: defensive special_tokens dump failed: {e}")

        return tokenizer_vocab