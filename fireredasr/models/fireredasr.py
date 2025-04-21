import os
import time

import torch

from fireredasr.data.asr_feat import ASRFeatExtractor
from fireredasr.models.fireredasr_aed import FireRedAsrAed
from fireredasr.models.fireredasr_llm import FireRedAsrLlm
from fireredasr.tokenizer.aed_tokenizer import ChineseCharEnglishSpmTokenizer
from fireredasr.tokenizer.llm_tokenizer import LlmTokenizerWrapper
from typing import Union, Optional


class FireRedAsr:
    @classmethod
    def from_pretrained(cls, asr_type, model_dir):
        assert asr_type in ["aed", "llm"]

        cmvn_path = os.path.join(model_dir, "cmvn.ark")
        feat_extractor = ASRFeatExtractor(cmvn_path)

        if asr_type == "aed":
            model_path = os.path.join(model_dir, "model.pth.tar")
            dict_path = os.path.join(model_dir, "dict.txt")
            spm_model = os.path.join(model_dir, "train_bpe1000.model")
            model = load_fireredasr_aed_model(model_path)
            tokenizer = ChineseCharEnglishSpmTokenizer(dict_path, spm_model)
        elif asr_type == "llm":
            model_path = os.path.join(model_dir, "model.pth.tar")
            encoder_path = os.path.join(model_dir, "asr_encoder.pth.tar")
            llm_dir = os.path.join(model_dir, "Qwen2-7B-Instruct")
            model, tokenizer = load_firered_llm_model_and_tokenizer(
                model_path, encoder_path, llm_dir)
        model.eval()
        return cls(asr_type, feat_extractor, model, tokenizer)

    def __init__(
        self,
        asr_type: str,
        feat_extractor: ASRFeatExtractor,
        model: Union[FireRedAsrAed, FireRedAsrLlm],
        tokenizer: Union[ChineseCharEnglishSpmTokenizer, LlmTokenizerWrapper],
    ):
        self.asr_type = asr_type
        self.feat_extractor = feat_extractor
        self.model = model
        self.tokenizer = tokenizer

    @torch.no_grad()
    def transcribe(self, batch_uttid, batch_wav_path, args={}):
        feats, lengths, durs = self.feat_extractor(batch_wav_path)
        total_dur = sum(durs)
        if args.get("use_gpu", False):
            feats, lengths = feats.cuda(), lengths.cuda()
            self.model.cuda()
        else:
            self.model.cpu()

        if self.asr_type == "aed":
            start_time = time.time()

            hyps = self.model.transcribe(
                feats, lengths,
                args.get("beam_size", 1),
                args.get("nbest", 1),
                args.get("decode_max_len", 0),
                args.get("softmax_smoothing", 1.0),
                args.get("aed_length_penalty", 0.0),
                args.get("eos_penalty", 1.0)
            )

            elapsed = time.time() - start_time
            rtf = elapsed / total_dur if total_dur > 0 else 0

            results = []
            for uttid, wav, hyp in zip(batch_uttid, batch_wav_path, hyps):
                hyp = hyp[0]  # only return 1-best
                hyp_ids = [int(id) for id in hyp["yseq"].cpu()]
                text = self.tokenizer.detokenize(hyp_ids)
                results.append({"uttid": uttid, "text": text, "wav": wav,
                                "rtf": f"{rtf:.4f}"})
            return results

        elif self.asr_type == "llm":
            input_ids, attention_mask, _, _ = \
                LlmTokenizerWrapper.preprocess_texts(
                    origin_texts=[""]*feats.size(0), tokenizer=self.tokenizer,
                    max_len=128, decode=True)
            if args.get("use_gpu", False):
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
            start_time = time.time()

            generated_ids = self.model.transcribe(
                feats, lengths, input_ids, attention_mask,
                args.get("beam_size", 1),
                args.get("decode_max_len", 0),
                args.get("decode_min_len", 0),
                args.get("repetition_penalty", 1.0),
                args.get("llm_length_penalty", 0.0),
                args.get("temperature", 1.0)
            )

            elapsed = time.time() - start_time
            rtf = elapsed / total_dur if total_dur > 0 else 0
            texts = self.tokenizer.batch_decode(generated_ids,
                                                skip_special_tokens=True)
            results = []
            for uttid, wav, text in zip(batch_uttid, batch_wav_path, texts):
                results.append({"uttid": uttid, "text": text, "wav": wav,
                                "rtf": f"{rtf:.4f}"})
            return results

    @torch.no_grad()
    def transcribe_tensor(self, batch_audio_tensor, sample_rate, batch_uttid=None, args={}):
        """
        直接从音频tensor进行转录，而不是从文件路径读取

        参数:
            batch_audio_tensor: 形状为 [batch_size, audio_length] 的音频波形tensor
            sample_rate: 音频采样率
            batch_uttid: 可选的音频ID列表，如果为None则使用随机ID
            args: 其他参数，与transcribe方法相同

        返回:
            转录结果列表
        """
        # 如果没有提供uttid，则生成随机ID
        if batch_uttid is None:
            batch_uttid = [f"utt_{i}" for i in range(len(batch_audio_tensor))]

        # 使用feat_extractor从tensor直接提取特征
        feats, lengths, durs = self.feat_extractor.extract_from_tensor(
            batch_audio_tensor, sample_rate)

        total_dur = sum(durs)
        if args.get("use_gpu", False):
            feats, lengths = feats.cuda(), lengths.cuda()
            self.model.cuda()
        else:
            self.model.cpu()

        if self.asr_type == "aed":
            start_time = time.time()

            hyps = self.model.transcribe(
                feats, lengths,
                args.get("beam_size", 1),
                args.get("nbest", 1),
                args.get("decode_max_len", 0),
                args.get("softmax_smoothing", 1.0),
                args.get("aed_length_penalty", 0.0),
                args.get("eos_penalty", 1.0)
            )

            elapsed = time.time() - start_time
            rtf = elapsed / total_dur if total_dur > 0 else 0

            results = []
            for uttid, hyp in zip(batch_uttid, hyps):
                hyp = hyp[0]  # only return 1-best
                hyp_ids = [int(id) for id in hyp["yseq"].cpu()]
                text = self.tokenizer.detokenize(hyp_ids)
                results.append({"uttid": uttid, "text": text,
                                "rtf": f"{rtf:.4f}"})
            return results

        elif self.asr_type == "llm":
            input_ids, attention_mask, _, _ = \
                LlmTokenizerWrapper.preprocess_texts(
                    origin_texts=[""]*feats.size(0), tokenizer=self.tokenizer,
                    max_len=128, decode=True)
            if args.get("use_gpu", False):
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
            start_time = time.time()

            generated_ids = self.model.transcribe(
                feats, lengths, input_ids, attention_mask,
                args.get("beam_size", 1),
                args.get("decode_max_len", 0),
                args.get("decode_min_len", 0),
                args.get("repetition_penalty", 1.0),
                args.get("llm_length_penalty", 0.0),
                args.get("temperature", 1.0)
            )

            elapsed = time.time() - start_time
            rtf = elapsed / total_dur if total_dur > 0 else 0
            texts = self.tokenizer.batch_decode(generated_ids,
                                                skip_special_tokens=True)
            results = []
            for uttid, text in zip(batch_uttid, texts):
                results.append({"uttid": uttid, "text": text,
                                "rtf": f"{rtf:.4f}"})
            return results


def load_fireredasr_aed_model(model_path):
    package = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    print("model args:", package["args"])
    model = FireRedAsrAed.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=True)
    return model


def load_firered_llm_model_and_tokenizer(model_path, encoder_path, llm_dir):
    package = torch.load(model_path, map_location=lambda storage, loc: storage, weights_only=False)
    package["args"].encoder_path = encoder_path
    package["args"].llm_dir = llm_dir
    print("model args:", package["args"])
    model = FireRedAsrLlm.from_args(package["args"])
    model.load_state_dict(package["model_state_dict"], strict=False)
    tokenizer = LlmTokenizerWrapper.build_llm_tokenizer(llm_dir)
    return model, tokenizer
