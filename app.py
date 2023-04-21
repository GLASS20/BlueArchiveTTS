import json
import os
import re

import librosa
import numpy as np
import torch
from torch import no_grad, LongTensor
import commons
import utils
import gradio as gr
from models import SynthesizerTrn
from text import text_to_sequence, _clean_text
from mel_processing import spectrogram_torch

limitation = os.getenv("SYSTEM") == "spaces"  # limit text and audio length in huggingface spaces


def get_text(text, hps, is_phoneme):
    text_norm = text_to_sequence(text, hps.symbols, [] if is_phoneme else hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = LongTensor(text_norm)
    return text_norm


def create_tts_fn(model, hps, speaker_ids):
    def tts_fn(text, speaker, speed, is_phoneme):
        if limitation:
            text_len = len(text)
            max_len = 500
            if is_phoneme:
                max_len *= 3
            else:
                if len(hps.data.text_cleaners) > 0 and hps.data.text_cleaners[0] == "zh_ja_mixture_cleaners":
                    text_len = len(re.sub("(\[ZH\]|\[JA\])", "", text))
            if text_len > max_len:
                return "Error: Text is too long", None

        speaker_id = speaker_ids[speaker]
        stn_tst = get_text(text, hps, is_phoneme)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0)
            x_tst_lengths = LongTensor([stn_tst.size(0)])
            sid = LongTensor([speaker_id])
            audio = model.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=.667, noise_scale_w=0.8,
                                length_scale=1.0 / speed)[0][0, 0].data.cpu().float().numpy()
        del stn_tst, x_tst, x_tst_lengths, sid
        return "Success", (hps.data.sampling_rate, audio)

    return tts_fn





def create_to_phoneme_fn(hps):
    def to_phoneme_fn(text):
        return _clean_text(text, hps.data.text_cleaners) if text != "" else ""

    return to_phoneme_fn


css = """
        #advanced-btn {
            color: white;
            border-color: black;
            background: black;
            font-size: .7rem !important;
            line-height: 19px;
            margin-top: 24px;
            margin-bottom: 12px;
            padding: 2px 8px;
            border-radius: 14px !important;
        }
        #advanced-options {
            display: none;
            margin-bottom: 20px;
        }
"""

if __name__ == '__main__':
    models_tts = []
    models_vc = []
    models_soft_vc = []
    name = 'BlueArchiveTTS'
    lang = '日本語 (Japanese)'
    example = '先生、何をお手伝いしましょうか？'
    config_path = f"saved_model/config.json"
    model_path = f"saved_model/model.pth"
    cover_path = f"saved_model/cover.png"
    hps = utils.get_hparams_from_file(config_path)
    model = SynthesizerTrn(
        len(hps.symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        n_speakers=hps.data.n_speakers,
        **hps.model)
    utils.load_checkpoint(model_path, model, None)
    model.eval()
    speaker_ids = [sid for sid, name in enumerate(hps.speakers) if name != "None"]
    speakers = [name for sid, name in enumerate(hps.speakers) if name != "None"]

    t = 'vits'
    models_tts.append((name, cover_path, speakers, lang, example,
                        hps.symbols, create_tts_fn(model, hps, speaker_ids),
                        create_to_phoneme_fn(hps)))
                               

    app = gr.Blocks(css=css)

    with app:
        gr.Markdown("# BlueArchiveTTS Using VITS Model\n\n"
                    "![visitor badge](https://visitor-badge.glitch.me/badge?page_id=ORI-Muchim.BlueArchiveTTS)\n\n")
        with gr.Tabs():
            with gr.TabItem("TTS"):
                with gr.Tabs():
                    for i, (name, cover_path, speakers, lang, example, symbols, tts_fn,
                            to_phoneme_fn) in enumerate(models_tts):
                        with gr.TabItem(f"BlueArchive"):
                            with gr.Column():
                                gr.Markdown(f"## {name}\n\n"
                                            f"![cover](file/{cover_path})\n\n"
                                            f"lang: {lang}")
                                tts_input1 = gr.TextArea(label="Text (500 words limitation)", value=example,
                                                         elem_id=f"tts-input{i}")
                                tts_input2 = gr.Dropdown(label="Speaker", choices=speakers,
                                                         type="index", value=speakers[0])
                                tts_input3 = gr.Slider(label="Speed", value=1, minimum=0.1, maximum=2, step=0.1)
                                with gr.Accordion(label="Advanced Options", open=False):
                                    phoneme_input = gr.Checkbox(value=False, label="Phoneme input")
                                    to_phoneme_btn = gr.Button("Covert text to phoneme")
                                    phoneme_list = gr.Dataset(label="Phoneme list", components=[tts_input1],
                                                              samples=[[x] for x in symbols],
                                                              elem_id=f"phoneme-list{i}")
                                    phoneme_list_json = gr.Json(value=symbols, visible=False)
                                tts_submit = gr.Button("Generate", variant="primary")
                                tts_output1 = gr.Textbox(label="Output Message")
                                tts_output2 = gr.Audio(label="Output Audio")
                                tts_submit.click(tts_fn, [tts_input1, tts_input2, tts_input3, phoneme_input],
                                                 [tts_output1, tts_output2])
                                to_phoneme_btn.click(to_phoneme_fn, [tts_input1], [tts_input1])
                                phoneme_list.click(None, [phoneme_list, phoneme_list_json], [],
                                                   _js=f"""
                                (i,phonemes) => {{
                                    let root = document.querySelector("body > gradio-app");
                                    if (root.shadowRoot != null)
                                        root = root.shadowRoot;
                                    let text_input = root.querySelector("#tts-input{i}").querySelector("textarea");
                                    let startPos = text_input.selectionStart;
                                    let endPos = text_input.selectionEnd;
                                    let oldTxt = text_input.value;
                                    let result = oldTxt.substring(0, startPos) + phonemes[i] + oldTxt.substring(endPos);
                                    text_input.value = result;
                                    let x = window.scrollX, y = window.scrollY;
                                    text_input.focus();
                                    text_input.selectionStart = startPos + phonemes[i].length;
                                    text_input.selectionEnd = startPos + phonemes[i].length;
                                    text_input.blur();
                                    window.scrollTo(x, y);
                                    return [];
                                }}""")
    
    app.queue(concurrency_count=3).launch(show_api=False)
