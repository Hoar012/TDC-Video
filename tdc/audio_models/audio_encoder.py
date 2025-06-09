import torch
import torch.nn as nn
import torch.nn.functional as F

from .Qformer import BertConfig, BertLMHeadModel
from .modeling_whisper import WhisperModel
from .beats.BEATs import BEATsConfig, BEATs

class AudioEncoder(nn.Module):
    @classmethod
    def init_speech_Qformer(cls, num_query_token, speech_width, num_hidden_layers=2):
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = speech_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

    def __init__(
        self,
        whisper_path="",
        freeze_whisper=True,
        beats_path="",
        freeze_beats=True,

        use_speech_Qformer=True,
        num_speech_query_token=1,
        freeze_speech_QFormer=False,
        window_level_Qformer=True,
        second_per_window=0.333333,
        second_stride=0.333333,
        
        speech_llama_proj_model="",
        freeze_speech_llama_proj=False,
        llama_hidden_size=0,
    ):
        super().__init__()

        self.beats_path = beats_path
        self.use_speech_Qformer = use_speech_Qformer
        self.window_level_Qformer = window_level_Qformer
        self.second_per_window = second_per_window
        self.second_stride = second_stride

        # assert whisper_path
        # self.speech_encoder = WhisperModel.from_pretrained(whisper_path).encoder
        # self.ln_speech = nn.LayerNorm(self.speech_encoder.config.d_model)
        # if freeze_whisper:
        #     for name, param in self.speech_encoder.named_parameters():
        #         param.requires_grad = False
        #     self.speech_encoder.eval()

        if self.beats_path:
            beats_ckpt = torch.load(self.beats_path, map_location='cpu')
            beats_cfg = BEATsConfig(beats_ckpt['cfg'])
            beats_cfg.encoder_layerdrop = -1.0 # [modified] deepspeed layerdrop会卡住
            self.beats = BEATs(beats_cfg)
            self.beats.load_state_dict(beats_ckpt['model'])
            # self.ln_audio = nn.LayerNorm(self.beats.cfg.encoder_embed_dim)
            if freeze_beats:
                for name, param in self.beats.named_parameters():
                    param.requires_grad = False
                self.beats.eval()

    def _encode_auditory_feature(self, speech_embeds, audio_embeds=None):
        if self.use_speech_Qformer:
            speech_embeds = self.ln_speech(speech_embeds)
            if audio_embeds is not None:
                audio_embeds = self.ln_audio(audio_embeds)
                if audio_embeds.size(1) < speech_embeds.size(1):
                    audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
                elif audio_embeds.size(1) > speech_embeds.size(1):
                    speech_embeds = F.pad(speech_embeds, (0, 0, 0, audio_embeds.size(1) - speech_embeds.size(1)))
                speech_embeds = torch.cat((speech_embeds, audio_embeds), dim=-1)
            speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)

            if self.window_level_Qformer:
                B, T, C = speech_embeds.shape
                kernel = round(1500 * self.second_per_window / 30.0)
                stride = round(1500 * self.second_stride / 30.0)
                kernel = (1, kernel)
                stride = (1, stride)
                speech_embeds_tr = speech_embeds.transpose(1, 2).unsqueeze(2)
                speech_embeds_overlap = F.unfold(speech_embeds_tr, kernel_size=kernel, dilation=1, padding=0, stride=stride)
                _, _, L = speech_embeds_overlap.shape
                speech_embeds_overlap = speech_embeds_overlap.view(B, -1, kernel[1], L)
                speech_embeds_overlap = torch.permute(speech_embeds_overlap, [0, 3, 2, 1])
                speech_embeds = speech_embeds_overlap.reshape(-1, kernel[1], C)
                speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long, device=speech_embeds.device)

            query_tokens = self.speech_query_tokens.expand(speech_embeds.shape[0], -1, -1)
            query_output = self.speech_Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=speech_embeds,
                encoder_attention_mask=speech_atts,
                return_dict=True,
            )
            speech_embeds = self.speech_llama_proj(query_output.last_hidden_state)

            if self.window_level_Qformer:
                speech_embeds = speech_embeds.view(B, -1, speech_embeds.size(2)).contiguous()

            speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)
        else:
            raise NotImplementedError

        return speech_embeds, speech_atts
    
    def _encode_auditory_feature(self, speech_embeds, audio_embeds=None):

        speech_embeds = self.ln_speech(speech_embeds)
        if audio_embeds is not None:
            audio_embeds = self.ln_audio(audio_embeds)
            if audio_embeds.size(1) < speech_embeds.size(1):
                audio_embeds = F.pad(audio_embeds, (0, 0, 0, speech_embeds.size(1) - audio_embeds.size(1)))
            elif audio_embeds.size(1) > speech_embeds.size(1):
                speech_embeds = F.pad(speech_embeds, (0, 0, 0, audio_embeds.size(1) - speech_embeds.size(1)))
            speech_embeds = torch.cat((speech_embeds, audio_embeds), dim=-1)
        speech_atts = torch.ones(speech_embeds.size()[:-1], dtype=torch.long).to(speech_embeds.device)

        return speech_embeds, speech_atts
    
    def encode_audio(self, audio_spectrogram, audio_wav=None, audio_wav_mask=None):
        speech_embeds = self.speech_encoder(audio_spectrogram, return_dict=True).last_hidden_state
        if self.beats_path and audio_wav is not None:
            audio_embeds, _ = self.beats.extract_features(audio_wav, padding_mask=audio_wav_mask, feature_only=True)
        else:
            audio_embeds = None

        return self._encode_auditory_feature(speech_embeds, audio_embeds=audio_embeds)
