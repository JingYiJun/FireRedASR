#!/usr/bin/env python3
"""
示例脚本：使用FireRedAsr模型直接处理音频张量
"""

import os
import torch
import torchaudio
import argparse

from fireredasr.models.fireredasr import FireRedAsr


def load_audio_tensor(audio_file: str):
    waveform, sample_rate = torchaudio.load(audio_file)

    # 重新采样到 16000
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        sample_rate = 16000

    # 将 float32 转换为 int16
    if waveform.dtype == torch.float32:
        waveform = (waveform * 32768.0).to(torch.int16)

    # 如果是多通道，只保留第一个通道
    if waveform.size(0) > 1:
        waveform = waveform[0:1]

    # 添加批处理维度，若已经有批处理维度则不需要
    if waveform.dim() == 2 and waveform.size(0) == 1:
        pass
    elif waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    return waveform, sample_rate


def main():
    parser = argparse.ArgumentParser(description="使用FireRedAsr模型直接处理音频张量")
    parser.add_argument("--model_dir", default="pretrained_models/FireRedASR-AED-L", help="模型目录")
    parser.add_argument("--asr_type", choices=["aed", "llm"], default="aed", help="ASR类型")
    parser.add_argument("--audio_file", default="data/test.wav", help="要转录的音频文件")
    parser.add_argument("--use_torchaudio", action="store_true", default=False, help="是否使用torchaudio加载音频")
    args = parser.parse_args()

    # 加载模型
    print(f"加载{args.asr_type}模型，路径：{args.model_dir}")
    model = FireRedAsr.from_pretrained(args.asr_type, args.model_dir)

    # 如果GPU可用，则使用GPU
    use_gpu = torch.cuda.is_available()

    if args.use_torchaudio:
        waveform, sample_rate = load_audio_tensor(args.audio_file)
        asr_args = {"use_gpu": use_gpu}
        results = model.transcribe_tensor(waveform, sample_rate, args=asr_args)
    else:
        asr_args = {"use_gpu": use_gpu}
        results = model.transcribe(["1"], [args.audio_file], args=asr_args)

    # 打印结果
    print("\n转录结果:")
    for res in results:
        print(f"ID: {res['uttid']}")
        print(f"文本: {res['text']}")
        print(f"RTF: {res['rtf']}")


if __name__ == "__main__":
    main()
