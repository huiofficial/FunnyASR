import gradio as gr
from funasr import AutoModel
import librosa
import logging
import numpy as np
from subtitle_utils import generate_srt


class ASRHandler():
    def __init__(self, funasr_model):
        self.funasr_model = funasr_model
        self.GLOBAL_COUNT = 0

    def recog(self, audio_input, sd_switch='no', state=None, hotwords=""):
        if state is None:
            state = {}
        sr, data = audio_input
        # assert sr == 16000, "16kHz sample rate required, {} given.".format(sr)
        if sr != 16000: # resample with librosa
            data = librosa.resample(data, orig_sr=sr, target_sr=16000)
        if len(data.shape) == 2:  # multi-channel wav input
            logging.warning("Input wav shape: {}, only first channel reserved.").format(data.shape)
            data = data[:,0]
        state['audio_input'] = (sr, data)
        data = data.astype(np.float64)
        if sd_switch == 'yes':
            rec_result = self.funasr_model.generate(data, return_raw_text=True, is_final=True, hotword=hotwords)
            res_srt = generate_srt(rec_result[0]['sentence_info'])
            state['sd_sentences'] = rec_result[0]['sentence_info']
        else:
            rec_result = self.funasr_model.generate(data,
                                                    return_spk_res=False,
                                                    sentence_timestamp=True,
                                                    return_raw_text=True,
                                                    is_final=True,
                                                    hotword=hotwords)
            res_srt = generate_srt(rec_result[0]['sentence_info'])
        state['recog_res_raw'] = rec_result[0]['raw_text']
        state['timestamp'] = rec_result[0]['timestamp']
        state['sentences'] = rec_result[0]['sentence_info']
        res_text = rec_result[0]['text']
        return res_text, res_srt, state


if __name__=="__main__":
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

    def audio_recog(audio_input, sd_switch, hotwords):
        # import pdb; pdb.set_trace()
        print(audio_input)
        return audio_clipper.recog(audio_input, sd_switch, hotwords=hotwords)

    with gr.Blocks() as demo:
        audio_state = gr.State()
        with gr.Tab("Audio Input"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(label="🔊音频输入 Audio Input")
                    with gr.Row():
                        audio_sd_switch = gr.Radio(["no", "yes"], label="👥是否区分说话人 Recognize Speakers", value='no')
                        hotwords_input2 = gr.Textbox(label="🚒热词 Hotwords")
                    recog_button1 = gr.Button("👂识别 Recognize")
                    audio_text_output = gr.Textbox(label="✏️识别结果 Recognition Result")
                    audio_srt_output = gr.Textbox(label="📖SRT字幕内容 RST Subtitles")

        recog_button1.click(audio_recog,
                            inputs=[audio_input, audio_sd_switch, hotwords_input2],
                            outputs=[audio_text_output, audio_srt_output, audio_state])

    demo.launch()

