"""Dataset loader with graceful HuggingFace fallback

This module provides smart dataset loading for MWS-Vision-Bench:
- Automatic download from HuggingFace (default)
- Graceful fallback if test set is not accessible
- Support for local files (development mode)
- Sample mode for quick testing

Copyright (c) 2024 MWS AI
Licensed under MIT License
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


def load_benchmark_datasets(
    data_paths: Optional[List[str]] = None,
    base_path: Optional[str] = None,
    hf_token: Optional[str] = None,
    cache_dir: Optional[str] = None,
    sample: Optional[int] = None,
    silent: bool = True
) -> Tuple[List[List[Dict[str, Any]]], List[str]]:
    """
    Load benchmark datasets with smart fallback logic.
    
    Priority:
    1. If data_paths provided → load local JSON files (requires base_path)
    2. Otherwise → download from HuggingFace with graceful fallback:
       - Always loads public validation set
       - Tries to load private test set (silently skips if no access)
    
    Args:
        data_paths: Local JSON file paths (if None, use HuggingFace)
        base_path: Base path for local images (required with data_paths)
        hf_token: HuggingFace token (for private test set access)
        cache_dir: Cache directory for HF datasets (default: ~/.cache/huggingface)
        sample: Number of samples to load (for quick testing)
        silent: If True, no warnings for missing test access
        
    Returns:
        Tuple of (datasets, split_names):
        - datasets: List of dataset lists (each is list of dicts)
        - split_names: Names of splits (e.g., ['validation', 'test'])
        
    Examples:
        # Load from HuggingFace (default)
        datasets, names = load_benchmark_datasets()
        
        # Load with test set (requires token)
        datasets, names = load_benchmark_datasets(hf_token="hf_...")
        
        # Quick test with 100 samples
        datasets, names = load_benchmark_datasets(sample=100)
        
        # Local files
        datasets, names = load_benchmark_datasets(
            data_paths=["./data/part1.json"],
            base_path="/path/to/images"
        )
    """
    
    # Local files mode
    if data_paths is not None:
        return _load_local_datasets(data_paths, base_path, sample)
    
    # HuggingFace mode (default)
    return _load_hf_datasets(hf_token, cache_dir, sample, silent)


def _load_local_datasets(
    data_paths: List[str],
    base_path: Optional[str],
    sample: Optional[int] = None
) -> Tuple[List[List[Dict[str, Any]]], List[str]]:
    """Load datasets from local JSON files.
    
    Args:
        data_paths: Paths to JSON files
        base_path: Base path for images
        sample: Number of samples to load per file
        
    Returns:
        Tuple of (datasets, split_names)
    """
    if not base_path or not os.path.exists(base_path):
        raise ValueError(
            "❌ Local data mode requires --base_path with images!\n"
            f"   Provided: {base_path}\n"
            f"   Files: {data_paths}"
        )
    
    datasets = []
    split_names = []
    
    for i, path in enumerate(data_paths):
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Data file not found: {path}")
        
        print(f"📁 Loading local file: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Apply sampling if requested
        if sample and sample < len(data):
            data = data[:sample]
            print(f"   ↳ Sampled {sample} examples from {len(data)} total")
        
        datasets.append(data)
        split_names.append(f"part{i+1}")
    
    print(f"✅ Loaded {len(datasets)} local dataset(s)")
    return datasets, split_names


def _load_hf_datasets(
    hf_token: Optional[str],
    cache_dir: Optional[str],
    sample: Optional[int],
    silent: bool
) -> Tuple[List[List[Dict[str, Any]]], List[str]]:
    """Load datasets from HuggingFace with graceful fallback.
    
    Args:
        hf_token: HuggingFace token for private datasets
        cache_dir: Cache directory
        sample: Number of samples to load
        silent: Suppress warnings
        
    Returns:
        Tuple of (datasets, split_names)
    """
    
    # Check if datasets library is available
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "❌ 'datasets' package not found!\n"
            "   Install it with: pip install datasets"
        )
    
    datasets = []
    split_names = []
    
    # Determine split string for sampling
    split_str = "train" if sample is None else f"train[:{sample}]"
    
    # 1. Load validation set (public - always works)
    print("📥 Downloading validation dataset from HuggingFace...")
    if sample:
        print(f"   ↳ Sampling first {sample} examples")
    
    try:
        val_dataset = load_dataset(
            "MTSAIR/MWS-Vision-Bench",
            split=split_str,
            cache_dir=cache_dir,
            token=hf_token
        )
        val_data = _convert_hf_to_list(val_dataset, split_name="validation")
        datasets.append(val_data)
        split_names.append("validation")
        print(f"✅ Validation dataset loaded ({len(val_data)} examples)")
    except Exception as e:
        raise RuntimeError(
            f"❌ Failed to load validation dataset from HuggingFace.\n"
            f"   Error: {e}\n"
            f"   Please check your internet connection.\n"
            f"   Dataset: https://huggingface.co/datasets/MTSAIR/MWS-Vision-Bench"
        )
    
    # 2. Try to load test set (private - graceful fallback)
    if hf_token:
        print("📥 Attempting to download test dataset (private)...")
        try:
            test_dataset = load_dataset(
                "MTSAIR/MWS-Vision-Bench-Test",
                split=split_str,
                cache_dir=cache_dir,
                token=hf_token
            )
            test_data = _convert_hf_to_list(test_dataset, split_name="test")
            datasets.append(test_data)
            split_names.append("test")
            print(f"✅ Test dataset loaded ({len(test_data)} examples)")
        except Exception as e:
            if not silent:
                print(f"ℹ️  Test dataset not accessible (this is normal for public users)")
                print(f"   Continuing with validation set only...")
    else:
        # No token provided - silently skip test set
        if not silent:
            print("ℹ️  No HF token provided - using validation set only")
            print("   Set HF_TOKEN environment variable to access test set")
    
    return datasets, split_names


def _convert_hf_to_list(hf_dataset, split_name: str = "default") -> List[Dict[str, Any]]:
    """Convert HuggingFace dataset to list of dictionaries with cached images.
    
    Args:
        hf_dataset: HuggingFace Dataset object
        split_name: Name of the split (validation/test) to avoid cache collisions
        
    Returns:
        List of dictionaries with dataset entries and local image paths
    """
    import tempfile
    import hashlib
    import io
    from PIL import Image
    
    data = []
    
    # Create cache directory for HF images with split name to avoid collisions
    # Different splits (validation/test) will have separate cache directories
    cache_dir = os.path.join(tempfile.gettempdir(), "mws_vision_bench_cache", split_name)
    os.makedirs(cache_dir, exist_ok=True)
    
    # Track image hashes to avoid saving duplicates
    image_hash_to_path = {}
    
    for idx, item in enumerate(hf_dataset):
        # Extract metadata (all fields are available directly in new format)
        entry = {
            "id": str(item.get("id", f"sample_{idx}")),
            "type": item.get("type", "unknown"),
            "dataset_name": item.get("dataset_name", "unknown"),
            "question": item.get("question", ""),
            "answers": item.get("answers", [])
        }
        
        # Handle HF Image type - save to cache and use local path
        if "image" in item:
            try:
                # HF Image object
                hf_image = item["image"]
                
                # Compute hash of the image to detect duplicates
                img_bytes_io = io.BytesIO()
                hf_image.save(img_bytes_io, format='PNG')
                img_bytes = img_bytes_io.getvalue()
                img_hash = hashlib.md5(img_bytes).hexdigest()
                
                # Check if we already saved this exact image
                if img_hash in image_hash_to_path:
                    # Reuse existing file path
                    entry["image_path"] = image_hash_to_path[img_hash]
                    data.append(entry)
                    continue
                
                # Determine file extension based on image mode
                if hf_image.mode in ('RGBA', 'LA', 'P'):
                    image_format = "PNG"
                    image_extension = ".png"
                else:
                    image_format = "JPEG"
                    image_extension = ".jpg"
                
                # Use hash as filename to ensure uniqueness and avoid duplicates
                image_filename = f"{img_hash}{image_extension}"
                image_path = os.path.join(cache_dir, image_filename)
                
                # Save image if not already cached
                if not os.path.exists(image_path):
                    # Convert RGBA to RGB for JPEG
                    if image_format == "JPEG" and hf_image.mode in ('RGBA', 'LA'):
                        # Create white background
                        rgb_image = Image.new('RGB', hf_image.size, (255, 255, 255))
                        if hf_image.mode == 'RGBA':
                            rgb_image.paste(hf_image, mask=hf_image.split()[3])  # Use alpha channel as mask
                        else:
                            rgb_image.paste(hf_image)
                        rgb_image.save(image_path, format=image_format)
                    else:
                        hf_image.save(image_path, format=image_format)
                
                # Remember this hash -> path mapping
                image_hash_to_path[img_hash] = image_path
                entry["image_path"] = image_path
            except Exception as e:
                print(f"⚠️  Warning: Failed to save image for ID {entry['id']}: {e}")
                entry["image_path"] = ""
        else:
            # Fallback for local files mode
            entry["image_path"] = item.get("image_path", "")
        
        data.append(entry)
    
    print(f"💾 Cached {len(data)} images to: {cache_dir}")
    return data


def get_dataset_info(
    data_paths: Optional[List[str]] = None,
    hf_token: Optional[str] = None
) -> Dict[str, Any]:
    """Get information about available datasets without loading them.
    
    Args:
        data_paths: Local file paths to check
        hf_token: HuggingFace token for checking private access
        
    Returns:
        Dictionary with dataset availability information
    """
    info = {
        "validation_available": False,
        "test_available": False,
        "source": None,
        "message": ""
    }
    
    # Check local files
    if data_paths:
        info["source"] = "local"
        info["validation_available"] = all(os.path.exists(p) for p in data_paths)
        info["test_available"] = len(data_paths) >= 2 if data_paths else False
        info["message"] = f"Local files: {data_paths}"
        return info
    
    # Check HuggingFace access
    info["source"] = "huggingface"
    info["validation_available"] = True  # Public dataset
    
    if hf_token:
        try:
            from datasets import load_dataset
            # Try to access test dataset
            load_dataset("MTSAIR/MWS-Vision-Bench-Test", split="train[:1]", token=hf_token)
            info["test_available"] = True
            info["message"] = "Full access: validation + test"
        except:
            info["test_available"] = False
            info["message"] = "Public access: validation only"
    else:
        info["test_available"] = False
        info["message"] = "Public access: validation only (set HF_TOKEN for test set)"
    
    return info

