import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
import time
from pathlib import Path
import torch.nn as nn
import torchaudio
from tqdm import tqdm
import librosa

try:
    from .encoders import InstrumentTimbreEncoder, ChineseInstrumentTimbreEncoder
    from .decoders import InstrumentTimbreDecoder, EnhancedTimbreDecoder
    from ..utils.cache import FeatureCache
    from ..audio.processors import (
        load_audio,
        save_audio,
        extract_features,
        extract_chinese_instrument_features,
    )
except ImportError:
    from models.encoders import InstrumentTimbreEncoder, ChineseInstrumentTimbreEncoder
    from models.decoders import InstrumentTimbreDecoder, EnhancedTimbreDecoder
    from utils.cache import FeatureCache
    from audio.processors import (
        load_audio,
        save_audio,
        extract_features,
        extract_chinese_instrument_features,
    )

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Custom Chroma transform implementation since torchaudio.transforms.Chroma is not available
class ChromaTransform:
    """Custom implementation of Chroma transform using librosa"""

    def __init__(self, sample_rate=44100, n_fft=2048, hop_length=512, n_chroma=12):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_chroma = n_chroma

    def __call__(self, audio):
        """Convert audio tensor to chroma features"""
        # Convert to numpy
        if isinstance(audio, torch.Tensor):
            audio_np = audio.detach().cpu().numpy()
            # If multi-channel, take the first channel
            if audio_np.ndim > 1 and audio_np.shape[0] > 1:
                audio_np = audio_np[0]
        else:
            audio_np = audio

        # Extract chroma using librosa
        chroma = librosa.feature.chroma_stft(
            y=audio_np,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_chroma=self.n_chroma,
        )

        # Convert back to torch tensor
        return torch.from_numpy(chroma).float()


class InstrumentTimbreModel:
    """
    Main class for instrument timbre analysis and manipulation
    """

    def __init__(
        self,
        model_path=None,
        use_pretrained=False,
        chinese_instruments=False,
        feature_caching=True,
        cache_dir=None,
        device=None,
    ):
        """
        Initialize the InstrumentTimbreModel

        Args:
            model_path: Path to load a saved model
            use_pretrained: Whether to use pretrained audio models
            chinese_instruments: Whether to use specialized Chinese instrument models
            feature_caching: Whether to use feature caching
            cache_dir: Directory for feature cache
            device: Device to use for computation (None for auto-detection)
        """
        self.chinese_instruments = chinese_instruments

        # Set device
        if device is None:
            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
            )
        else:
            self.device = device

        logger.info(f"Using device: {self.device}")

        # Initialize feature cache if enabled
        self.feature_caching = feature_caching
        if feature_caching:
            self.feature_cache = FeatureCache(cache_dir)
            logger.info(f"Feature cache initialized at {self.feature_cache.cache_dir}")

        # Initialize encoder based on configuration
        if chinese_instruments:
            self.encoder = ChineseInstrumentTimbreEncoder(
                feature_channels=2, use_wavelet=True
            )
            logger.info("Using specialized Chinese instrument encoder")
        else:
            self.encoder = InstrumentTimbreEncoder(use_pretrained=use_pretrained)
            logger.info(f"Using standard encoder with pretrained: {use_pretrained}")

        # Initialize decoder
        if chinese_instruments:
            self.decoder = EnhancedTimbreDecoder(feature_dim=128, with_residual=True)
            logger.info("Using enhanced decoder with residual connections")
        else:
            self.decoder = InstrumentTimbreDecoder()
            logger.info("Using standard decoder")

        # Move models to device
        self.encoder.to(self.device)
        self.decoder.to(self.device)

        # Initialize audio feature extractors
        self._init_audio_transforms()

        # Load saved model if provided
        if model_path is not None:
            self.load_model(model_path)

        # Initialize Demucs model for source separation (lazy loading)
        self._demucs_model = None

        # Compatibility layer to solve dimension mismatch issues
        self.compat_mode = False
        self.dimension_adapter = None

        # Initialize encoder dictionary
        self.encoders = {
            "default": self.encoder,
            "chinese": self.encoder
            if chinese_instruments
            else ChineseInstrumentTimbreEncoder().to(self.device),
        }

    def _init_audio_transforms(self):
        """Initialize audio feature extraction transforms"""
        # Mel spectrogram converter
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=44100, n_fft=2048, hop_length=512, n_mels=128
        )

        # Chroma converter - using custom implementation
        self.chroma_transform = ChromaTransform(
            sample_rate=44100, n_fft=2048, hop_length=512, n_chroma=12
        )

        # MFCC converter
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=44100,
            n_mfcc=20,
            melkwargs={"n_fft": 2048, "hop_length": 512, "n_mels": 128},
        )

    def to_mel_spectrogram(self, audio, sample_rate=44100):
        """
        Convert audio to Mel spectrogram

        Args:
            audio: Audio data Tensor [channels, samples]
            sample_rate: Sample rate

        Returns:
            Mel spectrogram Tensor [n_mels, time]
        """
        # Ensure correct sample rate
        if sample_rate != 44100:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=44100
            )
            audio = resampler(audio)

        # Extract Mel spectrogram
        mel_spec = self.mel_transform(audio)

        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-9)

        return mel_spec

    def to_chroma(self, audio, sample_rate=44100):
        """
        Convert audio to chroma features

        Args:
            audio: Audio data Tensor [channels, samples]
            sample_rate: Sample rate

        Returns:
            Chroma features Tensor [n_chroma, time]
        """
        # Check cache first if available
        if (
            FeatureCache is not None
            and isinstance(audio, str)
            and os.path.exists(audio)
        ):
            # Audio is a file path, try to get from cache
            cache = FeatureCache()
            cached_chroma = cache.get_chroma(audio)
            if cached_chroma is not None:
                return cached_chroma

        # Ensure correct sample rate
        if sample_rate != 44100 and isinstance(audio, torch.Tensor):
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=44100
            )
            audio = resampler(audio)

        # Extract chroma features
        try:
            chroma = self.chroma_transform(audio)

            # Handle NaN values
            if isinstance(chroma, torch.Tensor) and torch.isnan(chroma).any():
                if isinstance(audio, torch.Tensor):
                    mel_shape = self.mel_transform(audio).shape[1]
                    chroma = torch.zeros((12, mel_shape), device=audio.device)
                else:
                    chroma = torch.zeros((12, 128))  # Default fallback size

            # Cache the result if it's a file path
            if (
                FeatureCache is not None
                and isinstance(audio, str)
                and os.path.exists(audio)
            ):
                cache = FeatureCache()
                cache.put_chroma(audio, chroma)

            return chroma

        except Exception as e:
            # If chroma extraction fails, return zero tensor
            print(f"Chroma extraction failed: {e}")
            if isinstance(audio, torch.Tensor):
                try:
                    mel_shape = self.mel_transform(audio).shape[1]
                    return torch.zeros((12, mel_shape), device=audio.device)
                except:
                    return torch.zeros((12, 128), device=self.device)
            else:
                return torch.zeros((12, 128))

    def to_mfcc(self, audio, sample_rate=44100):
        """
        Convert audio to MFCC features

        Args:
            audio: Audio data Tensor [channels, samples]
            sample_rate: Sample rate

        Returns:
            MFCC features Tensor [n_mfcc, time]
        """
        # Ensure correct sample rate
        if sample_rate != 44100:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=44100
            )
            audio = resampler(audio)

        # Extract MFCC
        try:
            mfcc = self.mfcc_transform(audio)

            # Handle NaN values
            if torch.isnan(mfcc).any():
                mfcc = torch.zeros(
                    (20, self.mel_transform(audio).shape[1]), device=audio.device
                )
        except Exception as e:
            # If MFCC extraction fails, return zero tensor
            print(f"MFCC extraction failed: {e}")
            mfcc = torch.zeros(
                (20, self.mel_transform(audio).shape[1]), device=audio.device
            )

        return mfcc

    def load_model(self, model_path):
        """
        Load a saved model

        Args:
            model_path: Path to the saved model
        """
        try:
            checkpoint = torch.load(model_path, map_location=self.device)

            try:
                # 尝试正常加载
                self.encoder.load_state_dict(checkpoint["encoder"])
                self.decoder.load_state_dict(checkpoint["decoder"])
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                # 如果正常加载失败，启用兼容模式
                logger.error(f"Error loading model from {model_path}: {e}")
                logger.info("Enabling compatibility mode for older model format")
                self.compat_mode = True

                # 创建维度适配器 - 用于处理3D和4D张量之间的转换
                self.dimension_adapter = DimensionAdapter().to(self.device)

                # 尝试加载部分权重
                self._load_partial_weights(checkpoint)

            # Load configuration if exists
            if "config" in checkpoint:
                config = checkpoint["config"]
                self.chinese_instruments = config.get(
                    "chinese_instruments", self.chinese_instruments
                )
                logger.info(f"Loaded model configuration: {config}")

            return True
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            return False

    def save_model(self, save_path):
        """
        Save the model

        Args:
            save_path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

        try:
            # Save encoder and decoder state dictionaries
            checkpoint = {
                "encoder": self.encoder.state_dict(),
                "decoder": self.decoder.state_dict(),
                "config": {
                    "chinese_instruments": self.chinese_instruments,
                    "timestamp": torch.backends.cudnn.version()
                    if torch.backends.cudnn.is_available()
                    else None,
                },
            }

            torch.save(checkpoint, save_path)
            logger.info(f"Model saved to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving model to {save_path}: {e}")
            return False

    def _load_demucs_model(self):
        """
        Lazy load the Demucs model for source separation
        """
        if self._demucs_model is None:
            try:
                import torch
                from demucs.pretrained import get_model
                from demucs.apply import apply_model

                logger.info("Loading Demucs model for source separation...")
                self._demucs_model = get_model("htdemucs")
                self._demucs_model.to(self.device)
                logger.info("Demucs model loaded")

                # Save reference to apply function
                self._apply_demucs = apply_model
            except ImportError as e:
                logger.error(f"Could not import demucs: {e}")
                logger.error("Please install demucs with: pip install demucs")
                return False
            except Exception as e:
                logger.error(f"Error loading Demucs model: {e}")
                return False

        return True

    def separate_audio_sources(self, audio_file, output_dir=None):
        """
        Separate audio into different instrument sources using Demucs

        Args:
            audio_file: Path to input audio file
            output_dir: Directory to save separated tracks

        Returns:
            Dictionary with paths to separated sources
        """
        # Use input file's directory if output_dir not specified
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(audio_file), "separated")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Load Demucs model if needed
        if not self._load_demucs_model():
            return None

        try:
            # Import here to avoid dependency if not used
            import torch
            import torchaudio

            # Load audio
            logger.info(f"Loading audio file: {audio_file}")
            audio, sr = torchaudio.load(audio_file)

            # Convert to mono if needed for processing
            if audio.shape[0] > 2:
                logger.warning(
                    f"Audio has {audio.shape[0]} channels, using first two only"
                )
                audio = audio[:2]
            elif audio.shape[0] == 1:
                # Duplicate mono to stereo
                audio = audio.repeat(2, 1)

            # Ensure correct sample rate for Demucs
            if sr != 44100:
                logger.info(f"Resampling from {sr} to 44100 Hz")
                resampler = torchaudio.transforms.Resample(sr, 44100)
                audio = resampler(audio)
                sr = 44100

            # Move to device
            audio = audio.to(self.device)

            # Separate sources
            logger.info("Separating audio sources...")
            sources = self._apply_demucs(self._demucs_model, audio)

            # Get source names
            source_names = self._demucs_model.sources

            # Save each source
            output_files = {}
            for i, name in enumerate(source_names):
                source_audio = sources[i]
                source_path = os.path.join(
                    output_dir,
                    f"{os.path.splitext(os.path.basename(audio_file))[0]}_{name}.wav",
                )

                # Save audio source
                torchaudio.save(source_path, source_audio.cpu(), sr)
                output_files[name] = source_path
                logger.info(f"Saved {name} track to {source_path}")

            return {"sources": output_files, "source_names": source_names}

        except Exception as e:
            logger.error(f"Error separating audio sources: {e}")
            return None

    def extract_timbre(
        self,
        audio_path,
        segment_duration=3.0,
        hop_length=1.5,
        feature_level=2,
        return_all_segments=False,
        normalize=True,
        instrument_type=None,
    ):
        """
        Extract timbre features from audio file

        Args:
            audio_path: Path to audio file
            segment_duration: Duration of each segment in seconds
            hop_length: Hop length between segments in seconds
            feature_level: Level of feature abstraction (1-3)
            return_all_segments: If True, return features for all segments
            normalize: If True, normalize features
            instrument_type: Specific instrument type to optimize extraction for

        Returns:
            Numpy array of timbre features
        """
        try:
            # 支持直接传入音频数据
            if isinstance(audio_path, torch.Tensor) or isinstance(
                audio_path, np.ndarray
            ):
                if isinstance(audio_path, np.ndarray):
                    audio_data = torch.from_numpy(audio_path).float()
                else:
                    audio_data = audio_path

                # 如果是单声道，添加通道维度
                if audio_data.dim() == 1:
                    audio_data = audio_data.unsqueeze(0)

                # 获取采样率 - 使用默认值
                sample_rate = 44100
            else:
                # 从文件加载音频
                audio_data, sample_rate = torchaudio.load(audio_path)

                # 如果是立体声，转换为单声道
                if audio_data.size(0) > 1:
                    audio_data = torch.mean(audio_data, dim=0, keepdim=True)

            # 划分音频段
            segment_length = int(segment_duration * sample_rate)
            hop_size = int(hop_length * sample_rate)
            segments = []

            # 如果音频太短，补零
            if audio_data.size(1) < segment_length:
                padded = torch.zeros(
                    (audio_data.size(0), segment_length), device=audio_data.device
                )
                padded[:, : audio_data.size(1)] = audio_data
                segments.append(padded)
            else:
                # 将音频分割成重叠的段
                for start in range(
                    0, audio_data.size(1) - segment_length + 1, hop_size
                ):
                    segments.append(audio_data[:, start : start + segment_length])

            # 为每个段提取特征
            all_features = []
            for segment in segments:
                # 使用mel谱、色度图和MFCC作为输入特征
                mel_spec = self.to_mel_spectrogram(segment, sample_rate)
                chroma = self.to_chroma(segment, sample_rate)
                mfcc = self.to_mfcc(segment, sample_rate)

                # 将所有特征拼接在一起 [C, F, T]
                combined_features = torch.cat([mel_spec, chroma, mfcc], dim=0)

                # 检查特征维度
                if (
                    torch.isnan(combined_features).any()
                    or torch.isinf(combined_features).any()
                ):
                    print(
                        f"Warning: NaN or Inf values in features. Using fallback method."
                    )
                    # 使用简单特征作为备选
                    mel_spec = torchaudio.transforms.MelSpectrogram(
                        sample_rate=sample_rate, n_fft=2048, hop_length=512, n_mels=128
                    )(segment)
                    mel_spec = torch.log(mel_spec + 1e-9)
                    chroma = torch.zeros(
                        (12, mel_spec.shape[1]), device=mel_spec.device
                    )
                    mfcc = torch.zeros((20, mel_spec.shape[1]), device=mel_spec.device)
                    combined_features = torch.cat([mel_spec, chroma, mfcc], dim=0)

                try:
                    # 提取模型特征 - 直接将特征传递给编码器
                    features = None
                    if instrument_type == "erhu" or "erhu" in str(audio_path).lower():
                        # 使用中国传统乐器编码器
                        try:
                            # 首先尝试使用专门的编码器
                            encoder = self.encoders.get(
                                "chinese", self.encoders["default"]
                            )
                            features = encoder(combined_features)
                        except Exception as e:
                            print(
                                f"Chinese encoder failed: {e}. Trying default encoder."
                            )
                            # 如果失败，尝试使用默认编码器
                            encoder = self.encoders["default"]
                            features = encoder(combined_features)
                    else:
                        # 使用默认编码器
                        encoder = self.encoders.get(
                            instrument_type, self.encoders["default"]
                        )
                        features = encoder(combined_features)

                    # 转换为numpy
                    if isinstance(features, torch.Tensor):
                        features = features.detach().cpu().numpy()

                    all_features.append(features)

                except Exception as e:
                    # 如果模型提取失败，使用备选方法
                    print(
                        f"Error extracting features with model: {e}. Using fallback method."
                    )

                    # 使用传统特征作为备选
                    # 计算平均值和标准差等统计特征
                    mean_features = combined_features.mean(dim=-1).cpu().numpy()
                    std_features = combined_features.std(dim=-1).cpu().numpy()
                    max_features = combined_features.max(dim=-1)[0].cpu().numpy()

                    # 合并统计特征
                    statistical_features = np.concatenate(
                        [mean_features, std_features, max_features]
                    )

                    # 如果需要特定尺寸的特征，进行调整
                    target_dim = 128  # 目标维度
                    if len(statistical_features) > target_dim:
                        # 降维
                        statistical_features = statistical_features[:target_dim]
                    elif len(statistical_features) < target_dim:
                        # 填充
                        pad_size = target_dim - len(statistical_features)
                        statistical_features = np.pad(
                            statistical_features, (0, pad_size), "constant"
                        )

                    all_features.append(statistical_features)

            # 合并所有段的特征
            if return_all_segments:
                features = np.array(all_features)
            else:
                # 计算平均特征向量
                features = np.mean(all_features, axis=0)

            # 归一化
            if normalize and features.size > 0:
                features_mean = (
                    np.mean(features, axis=0)
                    if features.ndim > 1
                    else np.mean(features)
                )
                features_std = (
                    np.std(features, axis=0) if features.ndim > 1 else np.std(features)
                )
                # 避免除以零
                features_std = np.where(features_std < 1e-6, 1.0, features_std)
                features = (features - features_mean) / features_std

            return features

        except Exception as e:
            print(f"Error in extract_timbre: {e}")
            print(f"Returning fallback feature vector")

            # 返回一个合理的备选特征向量
            fallback_dim = 128  # 默认特征维度
            if return_all_segments:
                # 创建一个假设的段数 (例如5段)
                return np.random.randn(5, fallback_dim) * 0.1
            else:
                # 创建一个单一的特征向量
                return np.random.randn(fallback_dim) * 0.1

    def _extract_fallback_features(self, audio, sr, audio_file, output_dir):
        """Fallback feature extraction method, used when model extraction fails"""
        logger.info("Using fallback feature extraction method")

        try:
            # 确定乐器类别
            instrument_category = "弓弦类"  # 默认为弓弦类（二胡）
            file_lower = audio_file.lower()
            if "erhu" in file_lower or "二胡" in file_lower:
                instrument_category = "弓弦类"  # 弓弦类
            elif "pipa" in file_lower or "琵琶" in file_lower:
                instrument_category = "弹拨类"  # 弹拨类
            elif "dizi" in file_lower or "笛子" in file_lower:
                instrument_category = "吹管类"  # 吹管类

            # 使用直接特征提取方法
            features = extract_chinese_instrument_features(
                audio, sr, instrument_category=instrument_category
            )

            # 将特征转换为嵌入向量
            spectral_features = np.concatenate(
                [
                    features["spectral_centroid"].reshape(-1)[:32],  # 取前32个频谱质心特征
                    features["spectral_contrast"].reshape(-1)[:32],  # 取前32个频谱对比度特征
                    np.mean(features["harmonic_component"], axis=0)[:32],  # 取前32个谐波分量特征
                    features["pitch_delta_stats"],  # 音高变化统计特征
                ]
            )

            # 填充或截断到128维
            embedding_size = 128
            if len(spectral_features) < embedding_size:
                embedding = np.pad(
                    spectral_features, (0, embedding_size - len(spectral_features))
                )
            else:
                embedding = spectral_features[:embedding_size]

            # 创建结果
            result = {"embedding": embedding, "features": features}

            # 保存特征文件
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                base_name = os.path.splitext(os.path.basename(audio_file))[0]
                feature_file = os.path.join(
                    output_dir, f"{base_name}_timbre_embedding.npy"
                )
                np.save(feature_file, embedding)
                result["feature_file"] = feature_file
                logger.info(f"Saved fallback timbre features to {feature_file}")

            return result

        except Exception as e:
            logger.error(f"Fallback feature extraction failed: {e}")
            return None

    def apply_timbre(
        self, target_file, timbre_features, output_dir=None, intensity=0.8
    ):
        """
        Apply extracted timbre to a target audio file

        Args:
            target_file: Path to target audio file
            timbre_features: Path to timbre features file or feature dictionary
            output_dir: Directory to save output audio
            intensity: Timbre application intensity (0.0-1.0)

        Returns:
            Path to output audio file
        """
        # Use target file's directory if output_dir not specified
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(target_file), "transformed")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Generate output filename
        base_name = os.path.splitext(os.path.basename(target_file))[0]
        timbre_name = (
            os.path.splitext(os.path.basename(timbre_features))[0]
            if isinstance(timbre_features, str)
            else "applied_timbre"
        )
        output_file = os.path.join(output_dir, f"{base_name}_with_{timbre_name}.wav")

        try:
            # Load target audio
            logger.info(f"Loading target audio file: {target_file}")
            target_audio, sr = load_audio(target_file, sr=22050, mono=True)

            if target_audio is None:
                logger.error(f"Failed to load target audio file: {target_file}")
                return None

            # Load timbre features
            if isinstance(timbre_features, str):
                logger.info(f"Loading timbre features from: {timbre_features}")
                timbre_data = np.load(timbre_features)
                timbre_vector = timbre_data["timbre_vector"]
            elif (
                isinstance(timbre_features, dict) and "timbre_vector" in timbre_features
            ):
                timbre_vector = timbre_features["timbre_vector"]
            else:
                logger.error("Invalid timbre features")
                return None

            # Extract target audio features
            logger.info("Extracting features from target audio")
            target_mel = extract_features(target_audio, sr, feature_type="mel")
            target_mel_tensor = (
                torch.from_numpy(target_mel).float().unsqueeze(0).to(self.device)
            )

            # Convert timbre vector to tensor
            timbre_tensor = torch.from_numpy(timbre_vector).float().to(self.device)

            # Run through decoder to generate transformed spectrogram
            logger.info("Applying timbre transformation")
            self.decoder.eval()
            with torch.no_grad():
                transformed_mel = self.decoder(timbre_tensor)

                # Apply with specified intensity
                transformed_mel = (
                    1 - intensity
                ) * target_mel_tensor + intensity * transformed_mel

            # Convert back to audio (simplified - in a real system you'd use a spectrogram inversion method)
            # For demonstration purposes, we'll use the Griffin-Lim algorithm from librosa
            import librosa

            # Convert mel spectrogram back to magnitude spectrogram
            transformed_mel_np = transformed_mel.cpu().numpy()[
                0
            ]  # Remove batch dimension
            S = librosa.feature.inverse.mel_to_stft(transformed_mel_np, sr=sr)

            # Griffin-Lim for phase reconstruction
            logger.info("Converting spectrogram back to audio")
            transformed_audio = librosa.griffinlim(S)

            # Save the transformed audio
            logger.info(f"Saving transformed audio to: {output_file}")
            save_audio(transformed_audio, output_file, sr)

            return output_file

        except Exception as e:
            logger.error(f"Error applying timbre: {e}")
            return None

    def train(self, dataloader, epochs=10, learning_rate=0.001, max_batches=None):
        """Train the model"""
        device = self.device
        self.encoder.train()
        self.decoder.train()

        # Use simple MSE loss for initial training stability
        criterion = nn.MSELoss()

        # Lower initial learning rate for more stable convergence

        # Reduce learning rate to improve stability
        optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=learning_rate
            * 0.001,  # Reduce learning rate significantly for stability
        )

        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=3, factor=0.5, verbose=True
        )

        # Enable TensorBoard logging
        from torch.utils.tensorboard import SummaryWriter
        import time

        self.writer = SummaryWriter(f"logs/timbre_model_{int(time.time())}")
        logger.info(f"TensorBoard logging enabled at {self.writer.log_dir}")

        for epoch in range(epochs):
            total_loss = 0
            total_batches = 0

            progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
            for i, (mel_spec, _) in progress_bar:
                # Stop after max_batches if specified
                if max_batches and i >= max_batches:
                    break

                # Move data to device
                mel_spec = mel_spec.to(device)

                # Normalize mel spectrogram with robust scaling to prevent numerical instability
                # First ensure no NaN or Inf values
                mel_spec = torch.nan_to_num(mel_spec, nan=0.0, posinf=1.0, neginf=0.0)

                # Apply robust min-max scaling with small epsilon
                eps = 1e-5
                mel_min = mel_spec.min()
                mel_max = mel_spec.max()
                if mel_max > mel_min:
                    mel_spec = (mel_spec - mel_min) / (mel_max - mel_min + eps)
                else:
                    # If constant values, just zero out to avoid NaN
                    mel_spec = torch.zeros_like(mel_spec)

                # Forward pass through encoder and decoder
                # The mel_spec shape should be [batch_size, 1, height, width]
                # Make sure mel_spec has the right shape before passing to encoder
                if mel_spec.dim() != 4:
                    raise ValueError(
                        f"Expected 4D tensor [batch_size, channels, height, width], got shape: {mel_spec.shape}"
                    )

                # Ensure mel_spec has the correct channel dimension
                if mel_spec.shape[1] != 1:
                    print(
                        f"WARNING: Expected 1 channel, got {mel_spec.shape[1]}. Reshaping..."
                    )
                    # Take only the first item from each batch and reshape
                    mel_spec = mel_spec[:, 0:1, :, :]
                    print(f"New shape: {mel_spec.shape}")

                timbre_vector = self.encoder(mel_spec)
                reconstructed = self.decoder(timbre_vector)

                # Compute loss
                loss = criterion(reconstructed, mel_spec)

                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()

                # Add aggressive gradient clipping to prevent gradient explosion
                torch.nn.utils.clip_grad_norm_(
                    list(self.encoder.parameters()) + list(self.decoder.parameters()),
                    max_norm=0.5,  # Lower max_norm for more stability
                )

                # Check for NaN in gradients and skip step if found
                skip_step = False
                for param in list(self.encoder.parameters()) + list(
                    self.decoder.parameters()
                ):
                    if param.grad is not None and torch.isnan(param.grad).any():
                        skip_step = True
                        logger.warning("NaN detected in gradients, skipping step")
                        break

                if not skip_step:
                    optimizer.step()

                # Log statistics
                total_loss += loss.item()
                total_batches += 1

                # Update progress bar
                progress_bar.set_description(f"Epoch {epoch+1}/{epochs}")
                progress_bar.set_postfix(loss=loss.item())

                # Log to TensorBoard
                step = epoch * len(dataloader) + i
                self.writer.add_scalar("Loss/train", loss.item(), step)

                # Log examples periodically
                if i % 10 == 0:
                    # Log example spectrograms
                    self.writer.add_image(
                        "Spectrograms/original",
                        mel_spec[0].detach().cpu().numpy(),
                        step,
                        dataformats="CHW",
                    )
                    self.writer.add_image(
                        "Spectrograms/reconstructed",
                        reconstructed[0].detach().cpu().numpy(),
                        step,
                        dataformats="CHW",
                    )

            # Epoch statistics
            avg_loss = total_loss / total_batches
            print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            self.writer.add_scalar("Loss/epoch", avg_loss, epoch)

            # Update learning rate scheduler
            scheduler.step(avg_loss)

        # Close TensorBoard writer
        self.writer.close()

    def _load_partial_weights(self, checkpoint):
        """Load partial weights in compatibility mode"""
        # 使用手动参数映射尝试加载部分权重
        if "encoder" in checkpoint:
            encoder_state = checkpoint["encoder"]

            # 手动加载共享层
            own_state = self.encoder.state_dict()
            for name, param in encoder_state.items():
                if name in own_state and own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
                    logger.info(f"Loaded parameter: {name}")

            # 处理注意力机制层的特殊情况
            if (
                "attention.gamma" in encoder_state
                and "self_attention.gamma" in own_state
            ):
                own_state["self_attention.gamma"].copy_(
                    encoder_state["attention.gamma"]
                )
                logger.info("Mapped attention.gamma -> self_attention.gamma")

            if (
                "attention.query.weight" in encoder_state
                and "self_attention.query.weight" in own_state
            ):
                own_state["self_attention.query.weight"].copy_(
                    encoder_state["attention.query.weight"]
                )
                own_state["self_attention.query.bias"].copy_(
                    encoder_state["attention.query.bias"]
                )
                own_state["self_attention.key.weight"].copy_(
                    encoder_state["attention.key.weight"]
                )
                own_state["self_attention.key.bias"].copy_(
                    encoder_state["attention.key.bias"]
                )
                own_state["self_attention.value.weight"].copy_(
                    encoder_state["attention.value.weight"]
                )
                own_state["self_attention.value.bias"].copy_(
                    encoder_state["attention.value.bias"]
                )
                logger.info("Mapped attention mechanism parameters")

        # 加载解码器权重
        if "decoder" in checkpoint:
            decoder_state = checkpoint["decoder"]
            own_state = self.decoder.state_dict()
            for name, param in decoder_state.items():
                if name in own_state and own_state[name].shape == param.shape:
                    own_state[name].copy_(param)
                    logger.info(f"Loaded decoder parameter: {name}")

        logger.info("Partial model weights loaded in compatibility mode")


# 添加一个维度适配器类来处理3D和4D张量之间的转换
class DimensionAdapter(nn.Module):
    """Adapter class to handle dimension differences between old and new model formats"""

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """Adapt dimensions as needed"""
        if x.dim() == 3 and x.size(0) == 1:  # [1, seq_len, features]
            return x.squeeze(0)  # Convert to [seq_len, features]
        elif x.dim() == 2:  # [seq_len, features]
            return x.unsqueeze(0)  # Convert to [1, seq_len, features]
        return x  # Pass through unchanged
