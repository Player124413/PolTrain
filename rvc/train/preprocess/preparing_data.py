import os
import sys
import logging
import warnings

# Установка переменных окружения и отключение предупреждений
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")

import multiprocessing
import traceback
import time
from random import shuffle
import librosa
import numpy as np
import soundfile as sf
import torch
from fairseq.checkpoint_utils import load_model_ensemble_and_task
from fairseq.data.dictionary import Dictionary
from scipy import signal
from scipy.io import wavfile
from tqdm import tqdm

sys.path.append(os.getcwd())

from rvc.lib.audio import load_audio
from rvc.lib.rmvpe import RMVPE
from rvc.train.preprocess.slicer import Slicer

# Парсинг аргументов командной строки
exp_dir = str(sys.argv[1])
input_root = str(sys.argv[2])
embedder = str(sys.argv[3])
f0_method = str(sys.argv[4])
sample_rate = int(sys.argv[5])
percentage = float(sys.argv[6])
include_mutes = int(sys.argv[7])
normalize = sys.argv[8] == "True"

# Константы
RES_TYPE = "soxr_vhq"
SAMPLE_RATE_16K = 16000
num_processes = os.cpu_count()


class DataPreparer:
    """
    Класс для подготовки датасета для обучения RVC-модели.
    Объединяет логику сегментирования, ресемплинга, извлечения F0 и признаков HuBERT.
    """

    def __init__(self, exp_dir, input_root, percentage, sample_rate, normalize, embedder, f0_method, include_mutes):
        self.exp_dir = exp_dir
        self.input_root = input_root
        self.percentage = percentage
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.embedder = embedder
        self.f0_method = f0_method
        self.include_mutes = include_mutes

        # Настройка директорий
        self.gt_wavs_dir = os.path.join(exp_dir, "data", "sliced_audios")
        self.wavs16k_dir = os.path.join(exp_dir, "data", "sliced_audios_16k")
        self.f0_quant_path = os.path.join(exp_dir, "data", "f0_quantized")
        self.f0_voiced_path = os.path.join(exp_dir, "data", "f0_voiced")
        self.features_path = os.path.join(exp_dir, "data", "features")
        os.makedirs(self.gt_wavs_dir, exist_ok=True)
        os.makedirs(self.wavs16k_dir, exist_ok=True)
        os.makedirs(self.f0_quant_path, exist_ok=True)
        os.makedirs(self.f0_voiced_path, exist_ok=True)
        os.makedirs(self.features_path, exist_ok=True)

        # Параметры сегментирования аудио
        self.slicer = Slicer(sr=sample_rate, threshold=-42, min_length=1500, min_interval=400, hop_size=15, max_sil_kept=500)
        self.b_high, self.a_high = signal.butter(N=5, Wn=48, btype="high", fs=self.sample_rate)
        self.overlap = 0.3
        self.tail = self.percentage + self.overlap
        self.max_amplitude = 0.9
        self.alpha = 0.75

        # Параметры извлечения признаков
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.f0_bin = 256
        self.f0_min = 50.0
        self.f0_max = 1100.0
        self.f0_mel_min = 1127 * np.log(1 + self.f0_min / 700)
        self.f0_mel_max = 1127 * np.log(1 + self.f0_max / 700)

        # Инициализация моделей
        self.model_rmvpe = RMVPE(os.path.join(os.getcwd(), "rvc", "models", "predictors", "rmvpe.pt"), self.device)
        self.hubert_model = self._load_hubert_model()

    def _load_hubert_model(self):
        """Загрузка модели HuBERT."""
        torch.serialization.add_safe_globals([Dictionary])
        model_path = os.path.join(os.getcwd(), "rvc", "models", "embedders", self.embedder)
        models, _, _ = load_model_ensemble_and_task([model_path], suffix="")
        return models[0].to(self.device).eval()

    def _norm_write(self, tmp_audio, idx0, idx1):
        """Нормализация и сохранение аудио."""
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            return

        tmp_audio_resampled = librosa.resample(tmp_audio, orig_sr=self.sample_rate, target_sr=self.sample_rate, res_type=RES_TYPE)
        if self.normalize:
            tmp_audio_resampled = (tmp_audio_resampled / tmp_max * (self.max_amplitude * self.alpha)) + (1 - self.alpha) * tmp_audio_resampled
        wavfile.write(f"{self.gt_wavs_dir}/{idx0}_{idx1}.wav", self.sample_rate, tmp_audio_resampled.astype(np.float32))

        tmp_audio_16k = librosa.resample(tmp_audio_resampled, orig_sr=self.sample_rate, target_sr=SAMPLE_RATE_16K, res_type=RES_TYPE)
        wavfile.write(f"{self.wavs16k_dir}/{idx0}_{idx1}.wav", SAMPLE_RATE_16K, tmp_audio_16k.astype(np.float32))

    def _process_audio_file(self, path, idx0):
        """Обработка одного аудиофайла: сегментирование и сохранение."""
        try:
            audio = load_audio(path, self.sample_rate)
            audio = signal.lfilter(self.b_high, self.a_high, audio)
            idx1 = 0
            for audio in self.slicer.slice(audio):
                i = 0
                while True:
                    start = int(self.sample_rate * (self.percentage - self.overlap) * i)
                    i += 1
                    if len(audio[start:]) > self.tail * self.sample_rate:
                        tmp_audio = audio[start : start + int(self.percentage * self.sample_rate)]
                        self._norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                    else:
                        tmp_audio = audio[start:]
                        self._norm_write(tmp_audio, idx0, idx1)
                        idx1 += 1
                        break
        except Exception:
            print(f"Ошибка при обработке файла: {path}")
            traceback.print_exc()

    def _process_audio_chunk(self, infos, processed_count):
        """Часть работы для одного процесса с обновлением общего счетчика."""
        for path, idx0 in infos:
            self._process_audio_file(path, idx0)
            processed_count.value += 1

    def _run_multiprocessing(self, infos):
        """Запуск многопроцессорной обработки с индикатором прогресса."""
        manager = multiprocessing.Manager()
        processed_count = manager.Value('i', 0)
        total_files = len(infos)
        
        ps = [multiprocessing.Process(target=self._process_audio_chunk, args=(infos[i::num_processes], processed_count)) for i in range(num_processes)]
        
        for p in ps:
            p.start()
        
        with tqdm(total=total_files, desc="Сегментирование аудиофайлов") as pbar:
            while processed_count.value < total_files:
                pbar.update(processed_count.value - pbar.n)
                time.sleep(0.5)
            pbar.update(total_files - pbar.n)

        for p in ps:
            p.join()

    def _compute_f0(self, path):
        """Вычисление F0."""
        audio = load_audio(path, SAMPLE_RATE_16K)
        if self.f0_method == "rmvpe":
            return self.model_rmvpe.infer_from_audio(audio, 0.03)
        elif self.f0_method == "rmvpe+":
            return self.model_rmvpe.infer_from_audio_modified(audio, 0.02)

    def _coarse_f0(self, f0):
        """Квантование F0."""
        f0_mel = 1127 * np.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - self.f0_mel_min) * (self.f0_bin - 2) / (self.f0_mel_max - self.f0_mel_min) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > self.f0_bin - 1] = self.f0_bin - 1
        return np.rint(f0_mel).astype(int)

    def _read_wave(self, wav_path):
        """Чтение аудиофайла для HuBERT."""
        wav, sr = sf.read(wav_path)
        assert sr == SAMPLE_RATE_16K
        feats = torch.from_numpy(wav).float()
        if feats.dim() == 2:
            feats = feats.mean(-1)
        assert feats.dim() == 1
        return feats.view(1, -1)

    def _extract_features(self, wav_path):
        """Извлечение признаков HuBERT."""
        feats = self._read_wave(wav_path)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        with torch.no_grad():
            logits = self.hubert_model.extract_features(
                source=feats.to(self.device), padding_mask=padding_mask.to(self.device), output_layer=12
            )
            return logits[0].squeeze(0).float().cpu().numpy()

    def _extract_all_features(self):
        """Основной цикл извлечения F0 и признаков."""
        inp_root = self.wavs16k_dir
        files = sorted([f for f in os.listdir(inp_root) if f.endswith(".wav") and "spec" not in f])
        if not files:
            self._raise_no_files_error()

        print(f"\nФрагментов, готовых к обработке - {len(files)}")

        for file in tqdm(files, desc="Извлечение тона"):
            try:
                inp_path = f"{inp_root}/{file}"
                opt_path1 = f"{self.f0_quant_path}/{file}"
                opt_path2 = f"{self.f0_voiced_path}/{file}"
                if not (os.path.exists(opt_path1 + ".npy") and os.path.exists(opt_path2 + ".npy")):
                    featur_pit = self._compute_f0(inp_path)
                    np.save(opt_path2, featur_pit, allow_pickle=False)
                    coarse_pit = self._coarse_f0(featur_pit)
                    np.save(opt_path1, coarse_pit, allow_pickle=False)
            except Exception:
                raise RuntimeError(f"Ошибка извлечения тона!\nФайл - {inp_path}\n{traceback.format_exc()}")

        for file in tqdm(files, desc="Извлечение признаков"):
            try:
                wav_path = f"{inp_root}/{file}"
                out_path = f"{self.features_path}/{file.replace('.wav', '.npy')}"
                if not os.path.exists(out_path):
                    feats = self._extract_features(wav_path)
                    if np.isnan(feats).sum() > 0:
                        raise TypeError(f"Файл {file} содержит некорректные значения (NaN).")
                    np.save(out_path, feats, allow_pickle=False)
            except Exception:
                raise RuntimeError(f"Ошибка извлечения признаков!\nФайл - {wav_path}\n{traceback.format_exc()}")

    def _generate_filelist(self):
        """Генерация финального списка файлов (filelist.txt)."""
        mute_base_path = os.path.join(os.getcwd(), "rvc", "train", "preprocess", "mute")

        gt_wavs_files = set(name.split(".")[0] for name in os.listdir(self.gt_wavs_dir))
        feature_files = set(name.split(".")[0] for name in os.listdir(self.features_path))
        f0_files = set(name.split(".")[0] for name in os.listdir(self.f0_quant_path))
        f0nsf_files = set(name.split(".")[0] for name in os.listdir(self.f0_voiced_path))

        names = gt_wavs_files & feature_files & f0_files & f0nsf_files

        sids = []
        options = []
        for name in names:
            sid = name.split("_")[0]
            if sid not in sids:
                sids.append(sid)
            options.append(
                f"{os.path.join(self.gt_wavs_dir, name)}.wav|"
                f"{os.path.join(self.features_path, name)}.npy|"
                f"{os.path.join(self.f0_quant_path, name)}.wav.npy|"
                f"{os.path.join(self.f0_voiced_path, name)}.wav.npy|{sid}"
            )

        if self.include_mutes > 0:
            mute_audio_path = os.path.join(mute_base_path, "sliced_audios", f"mute{self.sample_rate}.wav")
            mute_feature_path = os.path.join(mute_base_path, "features", "mute.npy")
            mute_f0_path = os.path.join(mute_base_path, "f0_quantized", "mute.wav.npy")
            mute_f0nsf_path = os.path.join(mute_base_path, "f0_voiced", "mute.wav.npy")

            for sid in sids * self.include_mutes:
                options.append(f"{mute_audio_path}|{mute_feature_path}|{mute_f0_path}|{mute_f0nsf_path}|{sid}")

        shuffle(options)
        with open(os.path.join(self.exp_dir, "data", "filelist.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(options))

    def prepare_data(self):
        """Основной метод для запуска всего процесса подготовки данных."""
        print("\nСтарт предобработки датасета...\n")
        try:
            # 1. Сегментирование и ресемплинг
            infos = [(os.path.join(self.input_root, name), idx) for idx, name in enumerate(sorted(os.listdir(self.input_root)))]
            self._run_multiprocessing(infos)
            print("Сегментирование и ресемплинг успешно завершены!")

            # 2. Извлечение F0 и признаков
            self._extract_all_features()
            print("Извлечение признаков успешно завершено!")

            # 3. Генерация filelist.txt
            self._generate_filelist()
        except Exception as e:
            print(f"Критическая ошибка: {e}")
            print(traceback.format_exc())
            sys.exit(1)

    def _raise_no_files_error(self):
        error_message = (
            "ОШИБКА: Не найдено ни одного фрагмента для обработки.\n"
            "Возможные причины:\n"
            "1. Датасет не имеет звука.\n"
            "2. Датасет слишком тихий.\n"
            "3. Датасет слишком короткий (менее 3 секунд).\n"
            "4. Датасет слишком длинный (более 1 часа одним файлом).\n\n"
            "Попробуйте увеличить громкость или изменить объем датасета.\n"
            "Если у вас один большой файл, можно разделить его на несколько более мелких."
        )
        raise FileNotFoundError(error_message)


if __name__ == "__main__":
    preparer = DataPreparer(exp_dir, input_root, percentage, sample_rate, normalize, embedder, f0_method, include_mutes)
    preparer.prepare_data()
