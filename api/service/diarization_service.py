import os
import json
import wget
import uuid
from pydub import AudioSegment
import librosa
import soundfile
from omegaconf import OmegaConf
from nemo.collections.asr.models.msdd_models import NeuralDiarizer


class DiarizationService():
    def __init__(self, download_service):
        self.download_service = download_service

    def call_nemo(self, input_file_url):
        with self.download_service.download(input_file_url) as vocal_target:
            audio = AudioSegment.from_wav(vocal_target)
            chunk_length_ms = 60000  # 1 minute
            chunks = [audio[i:i+chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

            all_speaker_ts = []
            for idx, chunk in enumerate(chunks):
                chunk.export(f"chunk_{idx}.wav", format="wav")
                # Process each chunk as before
                signal, sample_rate = librosa.load(f"chunk_{idx}.wav", sr=None)
                soundfile.write(f"mono_chunk_{idx}.wav", signal, sample_rate, "PCM_24")
                
                _config = self.create_nemo_config(f"mono_chunk_{idx}.wav")
                model = NeuralDiarizer(cfg=_config)
                model.diarize()

                output_dir = "nemo_outputs"
                speaker_ts = []
                with open(f"{output_dir}/pred_rttms/mono_chunk_{idx}.rttm", "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        line_list = line.split(" ")
                        s = int(float(line_list[5]) * 1000) + idx * chunk_length_ms
                        e = s + int(float(line_list[8]) * 1000)
                        speaker_ts.append([s, e, int(line_list[11].split("_")[-1])])
                
                all_speaker_ts.extend(speaker_ts)

            # Open a file for writing
            file_name = 'nemo_results.json'
            with open(file_name, 'w') as f:
                json.dump(all_speaker_ts, f)

            return file_name

    # Helper function
    def create_nemo_config(self):
        data_dir = "./"
        DOMAIN_TYPE = "telephonic"  # Can be meeting or telephonic based on domain type of the audio file
        CONFIG_FILE_NAME = f"diar_infer_{DOMAIN_TYPE}.yaml"
        CONFIG_URL = f"https://raw.githubusercontent.com/NVIDIA/NeMo/main/examples/speaker_tasks/diarization/conf/inference/{CONFIG_FILE_NAME}"
        MODEL_CONFIG = os.path.join(data_dir, CONFIG_FILE_NAME)
        if not os.path.exists(MODEL_CONFIG):
            MODEL_CONFIG = wget.download(CONFIG_URL, data_dir)

        config = OmegaConf.load(MODEL_CONFIG)

        ROOT = os.getcwd()
        data_dir = os.path.join(ROOT, "data")
        os.makedirs(data_dir, exist_ok=True)

        meta = {
            "audio_filepath": "mono_file.wav",
            "offset": 0,
            "duration": None,
            "label": "infer",
            "text": "-",
            "rttm_filepath": None,
            "uem_filepath": None,
        }
        with open("data/input_manifest.json", "w") as fp:
            json.dump(meta, fp)
            fp.write("\n")

        pretrained_vad = "vad_multilingual_marblenet"
        pretrained_speaker_model = "titanet_large"

        config.num_workers = 0  # Workaround for multiprocessing hanging with ipython issue

        output_dir = "nemo_outputs"  # os.path.join(ROOT, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        config.diarizer.manifest_filepath = "data/input_manifest.json"
        config.diarizer.out_dir = (
            output_dir  # Directory to store intermediate files and prediction outputs
        )

        config.diarizer.speaker_embeddings.model_path = pretrained_speaker_model
        config.diarizer.oracle_vad = (
            False  # compute VAD provided with model_path to vad config
        )
        config.diarizer.clustering.parameters.oracle_num_speakers = False

        # Here, we use our in-house pretrained NeMo VAD model
        config.diarizer.vad.model_path = pretrained_vad
        config.diarizer.vad.parameters.onset = 0.8
        config.diarizer.vad.parameters.offset = 0.6
        config.diarizer.vad.parameters.pad_offset = -0.05
        config.diarizer.msdd_model.model_path = (
            "diar_msdd_telephonic"  # Telephonic speaker diarization model
        )

        return config