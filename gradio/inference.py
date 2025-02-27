
import os
import time
import torch
from utils import *
from config import *
from transformers import GPT2Config, LlamaConfig
from abctoolkit.utils import Exclaim_re, Quote_re, SquareBracket_re, Barline_regexPattern
from abctoolkit.transpose import Note_list, Pitch_sign_list
from abctoolkit.duration import calculate_bartext_duration

Note_list = Note_list + ['z', 'x']

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

patchilizer = Patchilizer()

patch_config = GPT2Config(num_hidden_layers=PATCH_NUM_LAYERS,
                          max_length=PATCH_LENGTH,
                          max_position_embeddings=PATCH_LENGTH,
                          n_embd=HIDDEN_SIZE,
                          num_attention_heads=HIDDEN_SIZE // 64,
                          vocab_size=1)
byte_config = GPT2Config(num_hidden_layers=CHAR_NUM_LAYERS,
                         max_length=PATCH_SIZE + 1,
                         max_position_embeddings=PATCH_SIZE + 1,
                         hidden_size=HIDDEN_SIZE,
                         num_attention_heads=HIDDEN_SIZE // 64,
                         vocab_size=128)

model = NotaGenLMHeadModel(encoder_config=patch_config, decoder_config=byte_config)

print("Parameter Number: " + str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

checkpoint = torch.load(INFERENCE_WEIGHTS_PATH, map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])
model = model.to(device)
model.eval()


def rest_unreduce(abc_lines):

    tunebody_index = None
    for i in range(len(abc_lines)):
        if '[V:' in abc_lines[i]:
            tunebody_index = i
            break

    metadata_lines = abc_lines[: tunebody_index]
    tunebody_lines = abc_lines[tunebody_index:]

    part_symbol_list = []
    voice_group_list = []
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
        except:
            pass    # use last ref_dur

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


def inference_patch(period, composer, instrumentation):

    prompt_lines = [
        '%' + period + '\n',
        '%' + composer + '\n',
        '%' + instrumentation + '\n']

    while True:

        failure_flag = False

        bos_patch = [patchilizer.bos_token_id] * (PATCH_SIZE - 1) + [patchilizer.eos_token_id]

        start_time = time.time()

        prompt_patches = patchilizer.patchilize_metadata(prompt_lines)
        byte_list = list(''.join(prompt_lines))
        print(''.join(byte_list), end='')

        prompt_patches = [[ord(c) for c in patch] + [patchilizer.special_token_id] * (PATCH_SIZE - len(patch)) for patch
                          in prompt_patches]
        prompt_patches.insert(0, bos_patch)

        input_patches = torch.tensor(prompt_patches, device=device).reshape(1, -1)

        end_flag = False
        cut_index = None

        tunebody_flag = False

        while True:
            predicted_patch = model.generate(input_patches.unsqueeze(0),
                                             top_k=TOP_K,
                                             top_p=TOP_P,
                                             temperature=TEMPERATURE)
            if not tunebody_flag and patchilizer.decode([predicted_patch]).startswith('[r:'):  # start with [r:0/
                tunebody_flag = True
                r0_patch = torch.tensor([ord(c) for c in '[r:0/']).unsqueeze(0).to(device)
                temp_input_patches = torch.concat([input_patches, r0_patch], axis=-1)
                predicted_patch = model.generate(temp_input_patches.unsqueeze(0),
                                                 top_k=TOP_K,
                                                 top_p=TOP_P,
                                                 temperature=TEMPERATURE)
                predicted_patch = [ord(c) for c in '[r:0/'] + predicted_patch
            if predicted_patch[0] == patchilizer.bos_token_id and predicted_patch[1] == patchilizer.eos_token_id:
                end_flag = True
                break
            next_patch = patchilizer.decode([predicted_patch])

            for char in next_patch:
                byte_list.append(char)
                print(char, end='')

            patch_end_flag = False
            for j in range(len(predicted_patch)):
                if patch_end_flag:
                    predicted_patch[j] = patchilizer.special_token_id
                if predicted_patch[j] == patchilizer.eos_token_id:
                    patch_end_flag = True

            predicted_patch = torch.tensor([predicted_patch], device=device)  # (1, 16)
            input_patches = torch.cat([input_patches, predicted_patch], dim=1)  # (1, 16 * patch_len)

            if len(byte_list) > 102400:
                failure_flag = True
                break
            if time.time() - start_time > 20 * 60:
                failure_flag = True
                break

            if input_patches.shape[1] >= PATCH_LENGTH * PATCH_SIZE and not end_flag:
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
                input_patches = patchilizer.encode_generate(abc_code_slice)

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


def inference_completetion(period, composer, instrumentation, start: str):
    prompt_lines = [
        '%' + period + '\n',
        '%' + composer + '\n',
        '%' + instrumentation + '\n']

    start = start.strip()

    while True:

        failure_flag = False

        bos_patch = [patchilizer.bos_token_id] * (PATCH_SIZE - 1) + [patchilizer.eos_token_id]

        start_time = time.time()

        prompt_patches = patchilizer.patchilize_metadata(prompt_lines)
        byte_list = list(''.join(prompt_lines))
        print(''.join(byte_list), end='')
        for s in start.split('\n'):
            print(s)

        prompt_patches = [[ord(c) for c in patch] + [patchilizer.special_token_id] * (PATCH_SIZE - len(patch)) for patch
                          in prompt_patches]
        prompt_patches.insert(0, bos_patch)

        start_patch = patchilizer.encode_generate(start)
        start_patch = [patch + [patchilizer.special_token_id] * (PATCH_SIZE - len(patch)) for patch in start_patch]
        input_patches = torch.tensor(prompt_patches + start_patch, device=device).reshape(1, -1)

        end_flag = False
        cut_index = None

        tunebody_flag = True

        while True:
            predicted_patch = model.generate(input_patches.unsqueeze(0),
                                             top_k=TOP_K,
                                             top_p=TOP_P,
                                             temperature=TEMPERATURE)
            if not tunebody_flag and patchilizer.decode([predicted_patch]).startswith('[r:'):  # start with [r:0/
                tunebody_flag = True
                r0_patch = torch.tensor([ord(c) for c in '[r:0/']).unsqueeze(0).to(device)
                temp_input_patches = torch.concat([input_patches, r0_patch], axis=-1)
                predicted_patch = model.generate(temp_input_patches.unsqueeze(0),
                                                 top_k=TOP_K,
                                                 top_p=TOP_P,
                                                 temperature=TEMPERATURE)
                predicted_patch = [ord(c) for c in '[r:0/'] + predicted_patch
            if predicted_patch[0] == patchilizer.bos_token_id and predicted_patch[1] == patchilizer.eos_token_id:
                end_flag = True
                break
            next_patch = patchilizer.decode([predicted_patch])

            for char in next_patch:
                byte_list.append(char)
                print(char, end='')

            patch_end_flag = False
            for j in range(len(predicted_patch)):
                if patch_end_flag:
                    predicted_patch[j] = patchilizer.special_token_id
                if predicted_patch[j] == patchilizer.eos_token_id:
                    patch_end_flag = True

            predicted_patch = torch.tensor([predicted_patch], device=device)  # (1, 16)
            input_patches = torch.cat([input_patches, predicted_patch], dim=1)  # (1, 16 * patch_len)

            # if len(byte_list) > 102400:
            #     failure_flag = True
            #     break
            # if time.time() - start_time > 20 * 60:
            #     failure_flag = True
            #     break

            if input_patches.shape[1] >= PATCH_LENGTH * PATCH_SIZE and not end_flag:
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
                input_patches = patchilizer.encode_generate(abc_code_slice)

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
            except Exception as e:
                print(e)
                failure_flag = True
                pass
            else:
                unreduced_abc_lines = [line for line in unreduced_abc_lines if not (line.startswith('%') and not line.startswith('%%'))]
                unreduced_abc_lines = ['X:1\n'] + unreduced_abc_lines
                unreduced_abc_text = ''.join(unreduced_abc_lines)
                return unreduced_abc_text

        assert failure_flag
        print("Failed to generate ABC notation. Restart me please")


if __name__ == '__main__':
    inference_completetion('Romantic', 'Chopin, Frederic', 'Keyboard', """
X:1
%%score { ( 1 4 ) | ( 2 3 5 ) }
L:1/8
Q:1/4=160
M:3/4
K:Am
V:1 treble nm="Piano" snm="Pno."
V:4 treble
V:2 bass
V:3 bass
V:5 bass
[r:0/192][V:1][I:staff +1] A,2|[V:2]x2|[V:3]x2|[V:4]x2|[V:5]x2|
[r:1/191][V:1][Q:1/4=160]!p![I:staff -1] z2[I:staff +1] E,C ^D,C|[V:2][A,,,A,,]2 z2 [A,,,A,,]2|[V:3]x6|[V:4]x6|[V:5]x6|
[r:2/190][V:1]=D,B, F,B,[I:staff -1] F2|[V:2][A,,,A,,]2 z2 [A,,D,]2|[V:3]x4 ^G,2|[V:4]x6|[V:5]x6|
[r:3/189][V:1]E2[I:staff +1] E,C ^D,C|[V:2][A,,,A,,]2 z2 [A,,,A,,]2|[V:3]A,2 x4|[V:4]x6|[V:5]x6|
[r:4/188][V:1]=D,B, F,B,[I:staff -1] F2|[V:2][A,,,A,,]2 z2 [A,,D,]2|[V:3]x4 ^G,2|[V:4]x6|[V:5]x6|
[r:5/187][V:1][I:staff +1] =D,B, F,B,[I:staff -1] F2|[V:2][A,,,A,,]2 z2 [A,,D,]2|[V:3]x2[I:staff -1] x2[I:staff +1] ^G,2|[V:4]x6|[V:5]x6|
[r:6/186][V:1]"_cresc."[I:staff +1] =D,B, F,B,[I:staff -1] F2|[V:2][A,,,A,,]2 z2 [A,,D,]2|[V:3]x2[I:staff -1] x2[I:staff +1] ^G,2|[V:4]x6|[V:5]x6|
[r:7/185][V:1]!fff! !>![bd'f']^a !>![bd']^^f !>![^gb]e|[V:2][A,,D,F,B,]2 [A,,D,F,B,]2 [A,,D,F,B,]2|[V:3]x6|[V:4]x6|[V:5]x6|
[r:8/184][V:1]!f! [^e^g]^c [d=f]^A [Bd]F|[V:2][A,,D,F,B,]2 [A,,D,F,B,]2 [A,,D,F,B,]2|[V:3]x6|[V:4]x6|[V:5]x6|
[r:9/183][V:1]!>!E4 A2|[V:2]A,,2 [E,C]2 [E,C]2|[V:3]x6|[V:4]x6|[V:5]x6|
[r:10/182][V:1](3(^GAG ^^FG AB|[V:2]A,,2 [E,D]2 [E,D]2|[V:3]x6|[V:4]x6|[V:5]x6|
[r:11/181][V:1]E4) c2|[V:2]A,,2 [E,C]2 [E,C]2|[V:3]x6|[V:4]x6|[V:5]x6|
[r:12/180][V:1]{Bc} B2 ^AB cd|[V:2]A,,2 [E,^G,D]2 [E,G,D]2|[V:3]x6|[V:4]x6|[V:5]x6|
[r:13/179][V:1]e2 (e'e e'^d|[V:2]A,,2 [E,C]2 [E,C]2|[V:3]x6|[V:4]x6|[V:5]x6|
[r:14/178][V:1]e'=d) (ec eB)|[V:2]A,,2 [E,^G,D]2 [E,G,D]2|[V:3]x6|[V:4]x6|[V:5]x6|
[r:15/177][V:1](3ABA ^GA d>c|[V:2]A,,2 [E,C]2 [E,C]2|[V:3]x6|[V:4]x6|[V:5]x6|
[r:16/176][V:1](Be EE){/G} F>E|[V:2]E,,2 [E,^G,D]2 [E,G,D]2|[V:3]x6|[V:4]x6|[V:5]x6|
[r:17/175][V:1]E4 A2|[V:2]A,,2 [E,C]2 [E,C]2|[V:3]x6|[V:4]x6|[V:5]x6|
[r:18/174][V:1](3(^GAG!<(! ^^FG AB)!<)!|[V:2]A,,2 [E,D]2 [E,D]2|[V:3]x6|[V:4]x6|[V:5]x6|
[r:19/173][V:1]z2 c4-|[V:2]A,,2 [E,C]2 _A,,2|[V:3]x6|[V:4]E4 F^F|[V:5]x6|
[r:20/172][V:1]cTB AB [Ac][Bd]|[V:2]G,,2 [G,D]2 [G,D]2|[V:3]x6|[V:4]G4 F2|[V:5]x6|
[r:21/171][V:1]e3 e e2|[V:2]z2 [E,G,C]2 [E,G,C]2|[V:3]x6|[V:4][EGc]6|[V:5]C,6|
[r:22/170][V:1]e3 e d2|[V:2]z2 B,4|[V:3]x6|[V:4]z2 F4|[V:5]D,6|
[r:23/169][V:1]cE ^Dc =DB|[V:2]z2 ^F,2 ^G,2|[V:3]x6|[V:4]x6|[V:5]{/E,,} E,6|
""")
