from funasr import AutoModel
from launch import ASRHandler
import logging
import librosa
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ASR Example')
    parser.add_argument('--input_file', type=str, default='test1.m4a', help='Input audio file')
    parser.add_argument('--output_text', type=str, default='output.txt', help='Output text file')
    parser.add_argument('--output_srt', type=str, default='output.srt', help='Output SRT file')
    parser.add_argument('--output_srt', type=str, default='output.srt', help='Output SRT file')
    args = parser.parse_args()

    funasr_model = AutoModel(model="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
                             model_revision="v2.0.4",
                             vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                             vad_model_revision="v2.0.4",
                             punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch",
                             punc_model_revision="v2.0.4",
                             spk_model="damo/speech_campplus_sv_zh-cn_16k-common",
                             spk_model_revision="v2.0.2",
                             )
    audio_clipper = ASRHandler(funasr_model)
    file = "test1.m4a"
    logging.warning("Recognizing audio file: {}".format(file))
    wav, sr = librosa.load(file, sr=16000)
    res_text, res_srt, state = audio_clipper.recog((sr, wav), 'no')
    print(res_text)
    print(res_srt)
    print(state)