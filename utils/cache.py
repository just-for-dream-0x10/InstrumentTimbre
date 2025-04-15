import os
import hashlib
import joblib
import numpy as np
import json
import torch
from pathlib import Path


class FeatureCache:
    """
    Cache for audio features to avoid recomputing features for the same files repeatedly
    """

    def __init__(self, cache_dir=None, max_cache_size=1000):
        """
        Initialize feature cache

        Args:
            cache_dir: Directory to store cache files, default is '~/.cache/instrument_timbre'
            max_cache_size: Maximum number of entries in the cache
        """
        if cache_dir is None:
            # Default cache directory in user's home directory
            cache_dir = os.path.expanduser("~/.cache/instrument_timbre")

        self.cache_dir = Path(cache_dir)
        self.max_cache_size = max_cache_size
        self.cache_index_file = self.cache_dir / "cache_index.json"
        self.cache_index = {}

        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)

        # Load cache index if it exists
        self._load_cache_index()

    def _load_cache_index(self):
        """Load the cache index from disk"""
        if os.path.exists(self.cache_index_file):
            try:
                with open(self.cache_index_file, "r") as f:
                    self.cache_index = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load cache index: {e}")
                self.cache_index = {}

    def _save_cache_index(self):
        """Save the cache index to disk"""
        try:
            with open(self.cache_index_file, "w") as f:
                json.dump(self.cache_index, f)
        except Exception as e:
            print(f"Warning: Could not save cache index: {e}")

    def _get_cache_key(self, file_path, feature_type, params):
        """
        Generate a unique cache key for a file and feature configuration

        Args:
            file_path: Path to the audio file
            feature_type: Type of feature extraction
            params: Additional parameters for feature extraction

        Returns:
            cache_key: Unique identifier for this file and feature configuration
        """
        # Get file modification time to invalidate cache when file changes
        try:
            mtime = os.path.getmtime(file_path)
        except:
            mtime = 0

        # Create a string with file path, mtime, feature type and params
        key_string = f"{file_path}_{mtime}_{feature_type}_{params}"

        # Generate MD5 hash of the key string
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, file_path, feature_type, params=None):
        """
        Get features from cache if available

        Args:
            file_path: Path to the audio file
            feature_type: Type of feature extraction
            params: Additional parameters for feature extraction

        Returns:
            features: Cached features or None if not in cache
        """
        if params is None:
            params = {}

        # Generate cache key
        cache_key = self._get_cache_key(file_path, feature_type, str(params))

        # Check if key exists in index
        if cache_key in self.cache_index:
            cache_file = self.cache_dir / f"{cache_key}.joblib"

            # Check if cache file exists
            if os.path.exists(cache_file):
                try:
                    # Load features from cache
                    features = joblib.load(cache_file)

                    # Convert to tensor if it's a numpy array
                    if isinstance(features, np.ndarray) and feature_type == "chroma":
                        features = torch.from_numpy(features).float()

                    # Update access timestamp
                    self.cache_index[cache_key]["last_access"] = os.path.getmtime(
                        file_path
                    )
                    self._save_cache_index()

                    return features
                except Exception as e:
                    print(f"Warning: Could not load cached features: {e}")
                    # Remove invalid entry
                    del self.cache_index[cache_key]
                    self._save_cache_index()

        return None

    def put(self, file_path, feature_type, features, params=None):
        """
        Store features in cache

        Args:
            file_path: Path to the audio file
            feature_type: Type of feature extraction
            features: Features to cache
            params: Additional parameters for feature extraction

        Returns:
            success: True if features were successfully cached
        """
        if params is None:
            params = {}

        # Generate cache key
        cache_key = self._get_cache_key(file_path, feature_type, str(params))

        # Create cache file path
        cache_file = self.cache_dir / f"{cache_key}.joblib"

        try:
            # Convert torch tensor to numpy for storage if needed
            if isinstance(features, torch.Tensor):
                features_to_save = features.detach().cpu().numpy()
            else:
                features_to_save = features

            # Save features to cache file
            joblib.dump(features_to_save, cache_file)

            # Update cache index
            self.cache_index[cache_key] = {
                "file_path": file_path,
                "feature_type": feature_type,
                "params": str(params),
                "last_access": os.path.getmtime(file_path),
                "size_bytes": os.path.getsize(cache_file),
            }

            self._save_cache_index()

            # Check if we need to prune the cache
            self._prune_cache_if_needed()

            return True
        except Exception as e:
            print(f"Warning: Could not cache features: {e}")
            return False

    def _prune_cache_if_needed(self):
        """Remove least recently used entries if cache size exceeds max_cache_size"""
        if len(self.cache_index) <= self.max_cache_size:
            return

        # Sort cache entries by last access time
        sorted_entries = sorted(
            self.cache_index.items(), key=lambda x: x[1].get("last_access", 0)
        )

        # Remove oldest entries until we're under the limit
        entries_to_remove = len(self.cache_index) - self.max_cache_size
        for i in range(entries_to_remove):
            key, entry = sorted_entries[i]
            cache_file = self.cache_dir / f"{key}.joblib"

            try:
                if os.path.exists(cache_file):
                    os.remove(cache_file)
                del self.cache_index[key]
            except Exception as e:
                print(f"Warning: Could not remove cache entry: {e}")

        self._save_cache_index()

    def clear(self):
        """Clear the entire cache"""
        try:
            # Remove all cache files
            for key in self.cache_index:
                cache_file = self.cache_dir / f"{key}.joblib"
                if os.path.exists(cache_file):
                    os.remove(cache_file)

            # Clear index
            self.cache_index = {}
            self._save_cache_index()

            print(f"Cache cleared: {self.cache_dir}")
            return True
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return False

    def stats(self):
        """Get cache statistics"""
        try:
            total_size = sum(
                entry.get("size_bytes", 0) for entry in self.cache_index.values()
            )
            return {
                "entries": len(self.cache_index),
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "max_entries": self.max_cache_size,
                "cache_dir": str(self.cache_dir),
            }
        except Exception as e:
            print(f"Error getting cache stats: {e}")
            return {}

    def get_chroma(self, file_path, params=None):
        """
        Specialized method to get cached chroma features

        Args:
            file_path: Path to the audio file
            params: Additional parameters for chroma extraction

        Returns:
            features: Cached chroma features or None if not in cache
        """
        return self.get(file_path, "chroma", params)

    def put_chroma(self, file_path, features, params=None):
        """
        Specialized method to cache chroma features

        Args:
            file_path: Path to the audio file
            features: Chroma features to cache
            params: Additional parameters for chroma extraction

        Returns:
            success: True if features were successfully cached
        """
        return self.put(file_path, "chroma", features, params)
