
import os
import time
import torch
from transformers import GPT2Config, LlamaConfig
from abctoolkit.utils import Exclaim_re, Quote_re, SquareBracket_re, Barline_regexPattern
from abctoolkit.transpose import Note_list, Pitch_sign_list
from abctoolkit.duration import calculate_bartext_duration
import torch
import random
import bisect
import json
import re
from transformers import GPT2Model, GPT2LMHeadModel, LlamaModel, LlamaForCausalLM, PreTrainedModel
from samplings import top_p_sampling, top_k_sampling, temperature_sampling
from tokenizers import Tokenizer
import tempfile
from tqdm import tqdm
import subprocess
from abctoolkit.transpose import Key2index, transpose_an_abc_text
from abctoolkit.check import check_alignment_unrotated
from abctoolkit.rotate import rotate_abc
from abctoolkit.convert import unidecode_abc_lines
from abctoolkit.utils import (
    remove_information_field,
    remove_bar_no_annotations,
    Quote_re,
    Barlines,
    extract_metadata_and_parts,
    extract_global_and_local_metadata,
    extract_barline_and_bartext_dict)

import sys
import typing

Note_list = Note_list + ['z', 'x']


class Patchilizer:
    def __init__(self, stream: bool, patch_size: int):
        self.stream = stream
        self.delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]
        self.regexPattern = '(' + '|'.join(map(re.escape, self.delimiters)) + ')'
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.special_token_id = 0
        self.patch_size = patch_size

    def split_bars(self, body_lines):
        """
        Split a body of music into individual bars.
        """
        new_bars = []
        try:
            for line in body_lines:
                line_bars = re.split(self.regexPattern, line)
                line_bars = list(filter(None, line_bars))
                new_line_bars = []

                if len(line_bars) == 1:
                    new_line_bars = line_bars
                else:
                    if line_bars[0] in self.delimiters:
                        new_line_bars = [line_bars[i] + line_bars[i + 1] for i in range(0, len(line_bars), 2)]
                    else:
                        new_line_bars = [line_bars[0]] + [line_bars[i] + line_bars[i + 1] for i in range(1, len(line_bars), 2)]
                    if 'V' not in new_line_bars[-1]:
                        new_line_bars[-2] += new_line_bars[-1]  # 吸收最后一个 小节线+\n 的组合
                        new_line_bars = new_line_bars[:-1]
                new_bars += new_line_bars
        except:
            pass

        return new_bars

    def split_patches(self, abc_text, patch_size: int, generate_last=False):
        if not generate_last and len(abc_text) % patch_size != 0:
            abc_text += chr(self.eos_token_id)
        patches = [abc_text[i: i + patch_size] for i in range(0, len(abc_text), patch_size)]
        return patches

    def patch2chars(self, patch):
        """
        Convert a patch into a bar.
        """
        bytes = ''
        for idx in patch:
            if idx == self.eos_token_id:
                break
            if idx < self.eos_token_id:
                pass
            bytes += chr(idx)
        return bytes

    def patchilize_metadata(self, metadata_lines):

        metadata_patches = []
        for line in metadata_lines:
            metadata_patches += self.split_patches(line, self.patch_size)

        return metadata_patches

    def patchilize_tunebody(self, tunebody_lines, encode_mode='train'):

        tunebody_patches = []
        bars = self.split_bars(tunebody_lines)
        if encode_mode == 'train':
            for bar in bars:
                tunebody_patches += self.split_patches(bar, self.patch_size)
        elif encode_mode == 'generate':
            for bar in bars[:-1]:
                tunebody_patches += self.split_patches(bar, self.patch_size)
            tunebody_patches += self.split_patches(bars[-1], self.patch_size, generate_last=True)

        return tunebody_patches

    def encode_train(self, abc_text, patch_length: bool, patch_size: int, add_special_patches=True, cut=True):

        lines = abc_text.split('\n')
        lines = list(filter(None, lines))
        lines = [line + '\n' for line in lines]

        tunebody_index = -1
        for i, line in enumerate(lines):
            if '[V:' in line:
                tunebody_index = i
                break

        metadata_lines = lines[: tunebody_index]
        tunebody_lines = lines[tunebody_index:]

        if self.stream:
            tunebody_lines = ['[r:' + str(line_index) + '/' + str(len(tunebody_lines) - line_index - 1) + ']' + line for line_index, line in
                              enumerate(tunebody_lines)]

        metadata_patches = self.patchilize_metadata(metadata_lines)
        tunebody_patches = self.patchilize_tunebody(tunebody_lines, encode_mode='train')

        if add_special_patches:
            bos_patch = chr(self.bos_token_id) * (patch_size - 1) + chr(self.eos_token_id)
            eos_patch = chr(self.bos_token_id) + chr(self.eos_token_id) * (patch_size - 1)

            metadata_patches = [bos_patch] + metadata_patches
            tunebody_patches = tunebody_patches + [eos_patch]

        if self.stream:
            if len(metadata_patches) + len(tunebody_patches) > patch_length:
                available_cut_indexes = [0] + [index + 1 for index, patch in enumerate(tunebody_patches) if '\n' in patch]
                line_index_for_cut_index = list(range(len(available_cut_indexes)))
                end_index = len(metadata_patches) + len(tunebody_patches) - patch_length
                biggest_index = bisect.bisect_left(available_cut_indexes, end_index)
                available_cut_indexes = available_cut_indexes[:biggest_index + 1]

                if len(available_cut_indexes) == 1:
                    choices = ['head']
                elif len(available_cut_indexes) == 2:
                    choices = ['head', 'tail']
                else:
                    choices = ['head', 'tail', 'middle']
                choice = random.choice(choices)
                if choice == 'head':
                    patches = metadata_patches + tunebody_patches[0:]
                else:
                    if choice == 'tail':
                        cut_index = len(available_cut_indexes) - 1
                    else:
                        cut_index = random.choice(range(1, len(available_cut_indexes) - 1))

                    line_index = line_index_for_cut_index[cut_index]
                    stream_tunebody_lines = tunebody_lines[line_index:]

                    stream_tunebody_patches = self.patchilize_tunebody(stream_tunebody_lines, encode_mode='train')
                    if add_special_patches:
                        stream_tunebody_patches = stream_tunebody_patches + [eos_patch]
                    patches = metadata_patches + stream_tunebody_patches
            else:
                patches = metadata_patches + tunebody_patches
        else:
            patches = metadata_patches + tunebody_patches

        if cut:
            patches = patches[: patch_length]
        else:
            pass

        # encode to ids
        id_patches = []
        for patch in patches:
            id_patch = [ord(c) for c in patch] + [self.special_token_id] * (patch_size - len(patch))
            id_patches.append(id_patch)

        return id_patches

    def encode_generate(self, abc_code: str, patch_length: int, patch_size: int, add_special_patches=True):

        lines = abc_code.split('\n')
        lines = list(filter(None, lines))

        tunebody_index = None
        for i, line in enumerate(lines):
            if line.startswith('[V:') or line.startswith('[r:'):
                tunebody_index = i
                break

        metadata_lines = lines[: tunebody_index]
        tunebody_lines = lines[tunebody_index:]

        metadata_lines = [line + '\n' for line in metadata_lines]
        if self.stream:
            if not abc_code.endswith('\n'):
                tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines) - 1)] + [tunebody_lines[-1]]
            else:
                tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines))]
        else:
            tunebody_lines = [line + '\n' for line in tunebody_lines]

        metadata_patches = self.patchilize_metadata(metadata_lines)
        tunebody_patches = self.patchilize_tunebody(tunebody_lines, encode_mode='generate')

        if add_special_patches:
            bos_patch = chr(self.bos_token_id) * (patch_size - 1) + chr(self.eos_token_id)

            metadata_patches = [bos_patch] + metadata_patches

        patches = metadata_patches + tunebody_patches
        patches = patches[: patch_length]

        # encode to ids
        id_patches = []
        for patch in patches:
            if len(patch) < patch_size and patch[-1] != chr(self.eos_token_id):
                id_patch = [ord(c) for c in patch]
            else:
                id_patch = [ord(c) for c in patch] + [self.special_token_id] * (patch_size - len(patch))
            id_patches.append(id_patch)

        return id_patches

    def decode(self, patches):
        """
        Decode patches into music.
        """
        return ''.join(self.patch2chars(patch) for patch in patches)


class PatchLevelDecoder(PreTrainedModel):
    """
    A Patch-level Decoder model for generating patch features in an auto-regressive manner.
    It inherits PreTrainedModel from transformers.
    """

    def __init__(self, config, patch_size: int):
        super().__init__(config)
        self.patch_embedding = torch.nn.Linear(patch_size * 128, config.n_embd)
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)
        self.base = GPT2Model(config)
        self.patch_size = patch_size

    def forward(self,
                patches: torch.Tensor,
                masks=None) -> torch.Tensor:
        """
        The forward pass of the patch-level decoder model.
        :param patches: the patches to be encoded
        :param masks: the masks for the patches
        :return: the encoded patches
        """
        patches = torch.nn.functional.one_hot(patches, num_classes=128).to(self.dtype)
        patches = patches.reshape(len(patches), -1, self.patch_size * (128))
        patches = self.patch_embedding(patches.to(self.device))

        if masks == None:
            return self.base(inputs_embeds=patches)
        else:
            return self.base(inputs_embeds=patches,
                             attention_mask=masks)


class CharLevelDecoder(PreTrainedModel):
    """
    A Char-level Decoder model for generating the chars within each patch in an auto-regressive manner
    based on the encoded patch features. It inherits PreTrainedModel from transformers.
    """

    def __init__(self, config, patch_sampling_batch_size: int):
        super().__init__(config)
        self.special_token_id = 0
        self.bos_token_id = 1

        self.base = GPT2LMHeadModel(config)
        self.patch_sampling_batch_size = patch_sampling_batch_size

    def forward(self,
                encoded_patches: torch.Tensor,
                target_patches: torch.Tensor):
        """
        The forward pass of the char-level decoder model.
        :param encoded_patches: the encoded patches
        :param target_patches: the target patches
        :return: the output of the model
        """
        # preparing the labels for model training
        target_patches = torch.cat((torch.ones_like(target_patches[:, 0:1])*self.bos_token_id, target_patches), dim=1)
        # print('target_patches shape:', target_patches.shape)

        target_masks = target_patches == self.special_token_id
        labels = target_patches.clone().masked_fill_(target_masks, -100)

        # masking the labels for model training
        target_masks = torch.ones_like(labels)
        target_masks = target_masks.masked_fill_(labels == -100, 0)

        # select patches
        if self.patch_sampling_batch_size != 0 and self.patch_sampling_batch_size < target_patches.shape[0]:
            indices = list(range(len(target_patches)))
            random.shuffle(indices)
            selected_indices = sorted(indices[:self.patch_sampling_batch_size])

            target_patches = target_patches[selected_indices, :]
            target_masks = target_masks[selected_indices, :]
            encoded_patches = encoded_patches[selected_indices, :]

        # get input embeddings
        inputs_embeds = torch.nn.functional.embedding(target_patches, self.base.transformer.wte.weight)

        # concatenate the encoded patches with the input embeddings
        inputs_embeds = torch.cat((encoded_patches.unsqueeze(1), inputs_embeds[:, 1:, :]), dim=1)

        output = self.base(inputs_embeds=inputs_embeds,
                           attention_mask=target_masks,
                           labels=labels)
        # output_hidden_states=True=True)

        return output

    def generate(
            self,
            encoded_patch: torch.Tensor,   # [hidden_size]
            tokens: torch.Tensor):  # [1]
        """
        The generate function for generating a patch based on the encoded patch and already generated tokens.
        :param encoded_patch: the encoded patch
        :param tokens: already generated tokens in the patch
        :return: the probability distribution of next token
        """
        encoded_patch = encoded_patch.reshape(1, 1, -1)  # [1, 1, hidden_size]
        tokens = tokens.reshape(1, -1)

        # Get input embeddings
        tokens = torch.nn.functional.embedding(tokens, self.base.transformer.wte.weight)

        # Concatenate the encoded patch with the input embeddings
        tokens = torch.cat((encoded_patch, tokens[:, 1:, :]), dim=1)

        # Get output from model
        outputs = self.base(inputs_embeds=tokens)

        # Get probabilities of next token
        probs = torch.nn.functional.softmax(outputs.logits.squeeze(0)[-1], dim=-1)

        return probs


class NotaGenLMHeadModel(PreTrainedModel):
    """
    NotaGen is a language model with a hierarchical structure.
    It includes a patch-level decoder and a char-level decoder.
    The patch-level decoder is used to generate patch features in an auto-regressive manner.
    The char-level decoder is used to generate the chars within each patch in an auto-regressive manner.
    It inherits PreTrainedModel from transformers.
    """

    def __init__(self, encoder_config, decoder_config, patch_size: int, patch_sampling_batch_size: int):
        super().__init__(encoder_config)
        self.special_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.patch_level_decoder = PatchLevelDecoder(encoder_config, patch_size)
        self.char_level_decoder = CharLevelDecoder(decoder_config, patch_sampling_batch_size)

    @property
    def patch_size(self):
        return self.patch_level_decoder.patch_size

    def forward(self,
                patches: torch.Tensor,
                masks: torch.Tensor):
        # """
        # The forward pass of the bGPT model.
        # :param patches: the patches to be encoded
        # :param masks: the masks for the patches
        # :return: the decoded patches
        # """
        # patches = patches.reshape(len(patches), -1, self.patch_size)
        # encoded_patches = self.patch_level_decoder(patches, masks)["last_hidden_state"]

        # left_shift_masks = masks * (masks.flip(1).cumsum(1).flip(1) > 1)
        # masks[:, 0] = 0

        # encoded_patches = encoded_patches[left_shift_masks == 1]
        # patches = patches[masks == 1]

        # return self.char_level_decoder(encoded_patches, patches)
        raise NotImplementedError

    def generate(self,
                 patches: torch.Tensor,
                 top_k=0,
                 top_p=1,
                 temperature=1.0):
        """
        The generate function for generating patches based on patches.
        :param patches: the patches to be encoded
        :param top_k: the top k for sampling
        :param top_p: the top p for sampling
        :param temperature: the temperature for sampling
        :return: the generated patches
        """
        if patches.shape[-1] % self.patch_size != 0:
            tokens = patches[:, :, -(patches.shape[-1] % self.patch_size):].squeeze(0, 1)
            tokens = torch.cat((torch.tensor([self.bos_token_id], device=self.device), tokens), dim=-1)
            patches = patches[:, :, :-(patches.shape[-1] % self.patch_size)]
        else:
            tokens = torch.tensor([self.bos_token_id], device=self.device)

        patches = patches.reshape(len(patches), -1, self.patch_size)  # [bs, seq, patch_size]
        encoded_patches = self.patch_level_decoder(patches)["last_hidden_state"]    # [bs, seq, hidden_size]
        generated_patch = []

        while True:
            prob = self.char_level_decoder.generate(encoded_patches[0][-1], tokens).cpu().detach().numpy()  # [128]
            prob = top_k_sampling(prob, top_k=top_k, return_probs=True)  # [128]
            prob = top_p_sampling(prob, top_p=top_p, return_probs=True)  # [128]
            token = temperature_sampling(prob, temperature=temperature)  # int
            char = chr(token)
            generated_patch.append(token)

            if len(tokens) >= self.patch_size:  # or token == self.eos_token_id:
                break
            else:
                tokens = torch.cat((tokens, torch.tensor([token], device=self.device)), dim=0)

        return generated_patch


class NotaGenSongGenerator:
    def __init__(
        self,
        inference_weights_path='',
        num_samples=1000,
        top_k=9,
        top_p=0.9,
        temperature=1.2,
        original_output_folder='../output/original',
        interleaved_output_folder='../output/interleaved',
        patch_stream=True,
        patch_size=16,
        patch_length=1024,
        char_num_layers=6,
        patch_num_layers=20,
        hidden_size=1280,
        patch_sampling_batch_size=0,
    ):
        self.inference_weights_path = inference_weights_path
        self.num_samples = num_samples
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature
        self.original_output_folder = os.path.join(original_output_folder, os.path.splitext(os.path.split(inference_weights_path)[-1])[0] + '_k_' + str(top_k) + '_p_' + str(top_p) + '_temp_' + str(temperature))
        self.interleaved_output_folder = os.path.join(interleaved_output_folder, os.path.splitext(os.path.split(inference_weights_path)[-1])[0] + '_k_' + str(top_k) + '_p_' + str(top_p) + '_temp_' + str(temperature))
        self.patch_stream = patch_stream
        self.patch_size = patch_size
        self.patch_length = patch_length
        self.char_num_layers = char_num_layers
        self.patch_num_layers = patch_num_layers
        self.hidden_size = hidden_size
        self.patch_sampling_batch_size = patch_sampling_batch_size
        self.patchilizer = Patchilizer(patch_stream, self.patch_size)

    @property
    def model(self):
        if hasattr(self, '_model'):
            return self._model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        patch_config = GPT2Config(num_hidden_layers=self.patch_num_layers,
                                  max_length=self.patch_length,
                                  max_position_embeddings=self.patch_length,
                                  n_embd=self.hidden_size,
                                  num_attention_heads=self.hidden_size // 64,
                                  vocab_size=1)
        byte_config = GPT2Config(num_hidden_layers=self.char_num_layers,
                                 max_length=self.patch_size + 1,
                                 max_position_embeddings=self.patch_size + 1,
                                 hidden_size=self.hidden_size,
                                 num_attention_heads=self.hidden_size // 64,
                                 vocab_size=128)

        model = NotaGenLMHeadModel(
            encoder_config=patch_config,
            decoder_config=byte_config,
            patch_size=self.patch_size,
            patch_sampling_batch_size=self.patch_sampling_batch_size
        )

        print("Parameter Number: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

        checkpoint = torch.load(self.inference_weights_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model'])
        model = model.to(device)
        model.eval()
        self._model = model
        return model

    def _get_valid_combinations(self):
        with open('prompts.txt', 'r') as f:
            prompts = f.readlines()
        valid_combinations = {}
        for prompt in prompts:
            prompt = prompt.strip()
            parts = prompt.split('_')
            valid_combinations[(parts[0].lower(), parts[1].lower(), parts[2].lower())] = (parts[0], parts[1], parts[2])
        return valid_combinations

    def _check_combinations(self, period: str, composer: str, instrumentation: str):
        combs = self._get_valid_combinations()
        c = (period.lower(), composer.lower(), instrumentation.lower())
        if c not in combs:
            raise ValueError(f"Invalid combination of period, composer, and instrumentation: {period}, {composer}, {instrumentation}")
        return combs[c]

    def inference_completetion(self, period: str, composer: str, instrumentation: str, start: str, output_stream: typing.TextIO):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        period, composer, instrumentation = self._check_combinations(period, composer, instrumentation)
        prompt_lines = [
            '%' + period + '\n',
            '%' + composer + '\n',
            '%' + instrumentation + '\n']

        start = start.strip()

        while True:
            failure_flag = False
            bos_patch = [self.patchilizer.bos_token_id] * (self.patch_size - 1) + [self.patchilizer.eos_token_id]
            start_time = time.time()

            prompt_patches = self.patchilizer.patchilize_metadata(prompt_lines)
            byte_list = list(''.join(prompt_lines))
            output_stream.write(''.join(byte_list))
            if start:
                for s in start.split('\n'):
                    output_stream.write(s + "\n")

            prompt_patches = [[ord(c) for c in patch] + [self.patchilizer.special_token_id] * (self.patch_size - len(patch)) for patch
                              in prompt_patches]
            prompt_patches.insert(0, bos_patch)

            if start:
                start_patch = self.patchilizer.encode_generate(start, self.patch_length, self.patch_size, add_special_patches=False)
                start_patch = [patch + [self.patchilizer.special_token_id] * (self.patch_size - len(patch)) for patch in start_patch]
                prompt_patches = prompt_patches + start_patch
            input_patches = torch.tensor(prompt_patches, device=device).reshape(1, -1)

            end_flag = False
            cut_index = None

            tunebody_flag = False

            while True:
                predicted_patch = self.model.generate(
                    input_patches.unsqueeze(0),
                    top_k=self.top_k,
                    top_p=self.top_p,
                    temperature=self.temperature
                )
                if not tunebody_flag and self.patchilizer.decode([predicted_patch]).startswith('[r:'):  # start with [r:0/
                    tunebody_flag = True
                    r0_patch = torch.tensor([ord(c) for c in '[r:0/']).unsqueeze(0).to(device)
                    temp_input_patches = torch.concat([input_patches, r0_patch], dim=-1)
                    predicted_patch = self.model.generate(
                        temp_input_patches.unsqueeze(0),
                        top_k=self.top_k,
                        top_p=self.top_p,
                        temperature=self.temperature
                    )
                    predicted_patch = [ord(c) for c in '[r:0/'] + predicted_patch
                if predicted_patch[0] == self.patchilizer.bos_token_id and predicted_patch[1] == self.patchilizer.eos_token_id:
                    end_flag = True
                    break
                next_patch = self.patchilizer.decode([predicted_patch])

                for char in next_patch:
                    byte_list.append(char)
                    output_stream.write(char)

                patch_end_flag = False
                for j in range(len(predicted_patch)):
                    if patch_end_flag:
                        predicted_patch[j] = self.patchilizer.special_token_id
                    if predicted_patch[j] == self.patchilizer.eos_token_id:
                        patch_end_flag = True

                predicted_patch = torch.tensor([predicted_patch], device=device)  # (1, 16)
                input_patches = torch.cat([input_patches, predicted_patch], dim=1)  # (1, 16 * patch_len)

                if len(byte_list) > 102400:
                    failure_flag = True
                    break
                if time.time() - start_time > 20 * 60:
                    failure_flag = True
                    break

                if input_patches.shape[1] >= self.patch_length * self.patch_size and not end_flag:
                    print('Stream generating...')
                    abc_code = ''.join(byte_list)
                    abc_lines = abc_code.split('\n')

                    tunebody_index = None
                    for i, line in enumerate(abc_lines):
                        if line.startswith('[r:') or line.startswith('[V:'):
                            tunebody_index = i
                            break
                    if tunebody_index is None or tunebody_index == len(abc_lines) - 1:
                        break

                    metadata_lines = abc_lines[:tunebody_index]
                    tunebody_lines = abc_lines[tunebody_index:]

                    metadata_lines = [line + '\n' for line in metadata_lines]
                    if not abc_code.endswith('\n'):
                        tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines) - 1)] + [
                            tunebody_lines[-1]]
                    else:
                        tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines))]

                    if cut_index is None:
                        cut_index = len(tunebody_lines) // 2

                    abc_code_slice = ''.join(metadata_lines + tunebody_lines[-cut_index:])
                    input_patches = self.patchilizer.encode_generate(abc_code_slice, self.patch_length, patch_size=self.patch_size, add_special_patches=False)

                    input_patches = [item for sublist in input_patches for item in sublist]
                    input_patches = torch.tensor([input_patches], device=device)
                    input_patches = input_patches.reshape(1, -1)

            if not failure_flag:
                abc_text = ''.join(byte_list)

                # unreduce
                abc_lines = abc_text.split('\n')
                abc_lines = list(filter(None, abc_lines))
                abc_lines = [line + '\n' for line in abc_lines]
                try:
                    unreduced_abc_lines = rest_unreduce(abc_lines)
                except:
                    failure_flag = True
                    pass
                else:
                    unreduced_abc_lines = [line for line in unreduced_abc_lines if not (line.startswith('%') and not line.startswith('%%'))]
                    unreduced_abc_lines = ['X:1\n'] + unreduced_abc_lines
                    unreduced_abc_text = ''.join(unreduced_abc_lines)
                    return unreduced_abc_text

            assert failure_flag
            print("Failed to generate ABC notation. Restarting...")
            output_stream.flush()
            output_stream.seek(0)

    def inference(self, period: str, composer: str, instrumentation: str, output_stream: typing.TextIO = sys.stdout):
        return self.inference_completetion(period, composer, instrumentation, "", output_stream)


def rest_unreduce(abc_lines: list[str]):
    tunebody_index = None
    for i in range(len(abc_lines)):
        if '[V:' in abc_lines[i]:
            tunebody_index = i
            break

    metadata_lines = abc_lines[: tunebody_index]
    tunebody_lines = abc_lines[tunebody_index:]

    part_symbol_list = []
    voice_group_list = []
    existed_voices = []
    for line in metadata_lines:
        if line.startswith('%%score'):
            for round_bracket_match in re.findall(r'\((.*?)\)', line):
                voice_group_list.append(round_bracket_match.split())
            existed_voices = [item for sublist in voice_group_list for item in sublist]
        if line.startswith('V:'):
            symbol = line.split()[0]
            part_symbol_list.append(symbol)
            if symbol[2:] not in existed_voices:
                voice_group_list.append([symbol[2:]])
    z_symbol_list = []  # voices that use z as rest
    x_symbol_list = []  # voices that use x as rest
    for voice_group in voice_group_list:
        z_symbol_list.append('V:' + voice_group[0])
        for j in range(1, len(voice_group)):
            x_symbol_list.append('V:' + voice_group[j])

    part_symbol_list.sort(key=lambda x: int(x[2:]))

    unreduced_tunebody_lines = []

    for i, line in enumerate(tunebody_lines):
        unreduced_line = ''

        line = re.sub(r'^\[r:[^\]]*\]', '', line)

        pattern = r'\[V:(\d+)\](.*?)(?=\[V:|$)'
        matches = re.findall(pattern, line)

        line_bar_dict = {}
        for match in matches:
            key = f'V:{match[0]}'
            value = match[1]
            line_bar_dict[key] = value

        # calculate duration and collect barline
        dur_dict = {}
        for symbol, bartext in line_bar_dict.items():
            right_barline = ''.join(re.split(Barline_regexPattern, bartext)[-2:])
            bartext = bartext[:-len(right_barline)]
            try:
                bar_dur = calculate_bartext_duration(bartext)
            except:
                bar_dur = None
            if bar_dur is not None:
                if bar_dur not in dur_dict.keys():
                    dur_dict[bar_dur] = 1
                else:
                    dur_dict[bar_dur] += 1

        try:
            ref_dur = max(dur_dict, key=dur_dict.get)
        except Exception as e:
            raise NotImplementedError("Handling not properly implemented yet")

        if i == 0:
            prefix_left_barline = line.split('[V:')[0]
        else:
            prefix_left_barline = ''

        for symbol in part_symbol_list:
            if symbol in line_bar_dict.keys():
                symbol_bartext = line_bar_dict[symbol]
            else:
                if symbol in z_symbol_list:
                    symbol_bartext = prefix_left_barline + 'z' + str(ref_dur) + right_barline
                elif symbol in x_symbol_list:
                    symbol_bartext = prefix_left_barline + 'x' + str(ref_dur) + right_barline
            unreduced_line += '[' + symbol + ']' + symbol_bartext

        unreduced_tunebody_lines.append(unreduced_line + '\n')

    unreduced_lines = metadata_lines + unreduced_tunebody_lines

    return unreduced_lines


def convert_xml2abc(file: str):
    cmd = 'python xml2abc.py -d 8 -c 6 -x '
    try:
        p = subprocess.Popen(cmd + '"' + file + '"', stdout=subprocess.PIPE, shell=True)
        result = p.communicate()
        output = result[0].decode('utf-8')

        if output == '':
            raise ValueError(f"Conversion failed: {file}")
        return output
    except Exception as e:
        raise ValueError(file + ' ' + str(e) + '\n')


def interleave_abc(abc_str: str):
    abc_lines = abc_str.split('\n')
    abc_lines = [line for line in abc_lines if line.strip() != '']
    abc_lines = unidecode_abc_lines(abc_lines)

    # clean information field
    abc_lines = remove_information_field(abc_lines=abc_lines, info_fields=['X:', 'T:', 'C:', 'W:', 'w:', 'Z:', '%%MIDI'])

    # delete bar number annotations
    abc_lines = remove_bar_no_annotations(abc_lines)

    # delete \"
    for i, line in enumerate(abc_lines):
        if re.search(r'^[A-Za-z]:', line) or line.startswith('%'):
            continue
        else:
            if r'\"' in line:
                abc_lines[i] = abc_lines[i].replace(r'\"', '')

    # delete text annotations with quotes
    for i, line in enumerate(abc_lines):
        quote_contents = re.findall(Quote_re, line)
        for quote_content in quote_contents:
            for barline in Barlines:
                if barline in quote_content:
                    line = line.replace(quote_content, '')
                    abc_lines[i] = line

    # check bar alignment
    _, bar_no_equal_flag, _ = check_alignment_unrotated(abc_lines)
    if not bar_no_equal_flag:
        raise Exception('Unequal bar number')

    # deal with text annotations: remove too long text annotations; remove consecutive non-alphabet/number characters
    for i, line in enumerate(abc_lines):
        quote_matches = re.findall(r'"[^"]*"', line)
        for match in quote_matches:
            if match == '""':
                line = line.replace(match, '')
            if match[1] in ['^', '_']:
                sub_string = match
                pattern = r'([^a-zA-Z0-9])\1+'
                sub_string = re.sub(pattern, r'\1', sub_string)
                if len(sub_string) <= 40:
                    line = line.replace(match, sub_string)
                else:
                    line = line.replace(match, '')
        abc_lines[i] = line

    # transpose
    metadata_lines, part_text_dict = extract_metadata_and_parts(abc_lines)
    global_metadata_dict, local_metadata_dict = extract_global_and_local_metadata(metadata_lines)
    if global_metadata_dict['K'][0] == 'none':
        global_metadata_dict['K'][0] = 'C'

    interleaved_abc = rotate_abc(abc_lines)
    return interleaved_abc


def bake_target_nbars(interleaved_abc: list[str], target_nbars: int):
    i = 0
    baked_abc = []
    metadata = True
    for line in interleaved_abc:
        if not line.startswith("[V:"):
            baked_abc.append(line)
            continue
        metadata = False
        if not line.startswith("[V:1]"):
            raise ValueError(f"Invalid interleaved abc: {interleaved_abc}")
        if target_nbars-i <= 0:
            raise ValueError(f"Target nbars less than number of bars provided! ({target_nbars} is too small)")
        baked_abc.append(f"[r:{i}/{target_nbars-i}]")


def abc2xml(abctext: str, output_dir: str):
    with tempfile.NamedTemporaryFile("rw", suffix=".abc") as f:
        f.write(abctext.encode())
        cmd = ["../.venv/Scripts/python.exe", "abc2xml.py", '-o', output_dir, f.name]
        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            error_msg = f"Conversion failed: {e.stderr}" if e.stderr else "Unknown error"
            raise ValueError(error_msg)


if __name__ == '__main__':
    gen = NotaGenSongGenerator("gradio/weights_notagenx_p_size_16_p_length_1024_p_layers_20_h_size_1280.pth")
