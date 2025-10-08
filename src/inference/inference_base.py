"""
MWSVisionBench - Russian OCR benchmark for multimodal LLMs

This file: Base class for inference scripts with unified architecture and parameter validation

Copyright (c) 2024 MWS AI
Licensed under MIT License
"""

# Standard library imports
import argparse
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Any, Dict, List, Optional, Set


class InferenceBase(ABC):
    """Base class for all inference implementations"""
    
    # Define supported parameters for each API type
    SUPPORTED_PARAMS = {
        'openai': {'temperature', 'top_p', 'presence_penalty', 'frequency_penalty', 'max_tokens'},
        'gigachat': {'temperature', 'top_p', 'max_tokens', 'repetition_penalty'},  # Fixed based on official docs
        'responses': {'max_tokens'},  # GPT-5 has very limited params
        'local': {'temperature', 'top_p', 'max_tokens'}
    }
    
    def __init__(self, api_type: str):
        self.api_type = api_type
        self.supported_params = self.SUPPORTED_PARAMS.get(api_type, set())
        self.args = None
        self.data = None
        self.results = []
        self.processed_ids = set()
        self.error_ids = set()
        
    def setup_args(self):
        """Setup command line arguments - common for all inference scripts"""
        parser = argparse.ArgumentParser()
        parser.add_argument('--model_name', default=self.get_default_model())
        parser.add_argument('--api_url', default=self.get_default_api_url())
        parser.add_argument('--base_path', default=None)
        parser.add_argument('--api_key', default=None)
        parser.add_argument('--data_path', default="small_test.json")
        parser.add_argument('--start_index', type=int, default=0)
        parser.add_argument('--end_index', type=int, default=-1)
        parser.add_argument('--output_path', required=True, help='Path to save raw results')
        parser.add_argument('--use_base_prompt', action='store_true', help='Whether to use base prompt')
        parser.add_argument('--max_workers', type=int, help='Number of parallel workers')
        self.args = parser.parse_args()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[logging.StreamHandler()]
        )
        
    def validate_and_warn_params(self):
        """Check environment variables and warn about unsupported parameters"""
        env_params = {
            'TEMPERATURE': 'temperature',
            'TOP_P': 'top_p', 
            'PRESENCE_PENALTY': 'presence_penalty',
            'FREQUENCY_PENALTY': 'frequency_penalty',
            'MAX_TOKENS': 'max_tokens',
            'REPETITION_PENALTY': 'repetition_penalty',
            'SYSTEM_PROMPT': 'system_prompt'
        }
        
        warnings = []
        for env_var, param_name in env_params.items():
            if os.getenv(env_var) and param_name not in self.supported_params and param_name != 'system_prompt':
                warnings.append(f"{env_var} ({param_name})")
        
        if warnings:
            logging.warning(f"⚠️  The following parameters are set in environment but NOT SUPPORTED by {self.api_type} API: {', '.join(warnings)}")
            logging.warning("These parameters will be ignored. Continuing with supported parameters only.")
        
        # Log supported parameters
        if self.supported_params:
            supported_env = [env_var for env_var, param_name in env_params.items() 
                           if param_name in self.supported_params or param_name == 'system_prompt']
            
            logging.info(f"✅ {self.api_type.upper()} API supports: {', '.join(supported_env)}")
    
    def get_supported_params(self) -> Dict[str, Any]:
        """Get environment parameters that are supported by this API"""
        params = {}
        
        if 'temperature' in self.supported_params:
            params['temperature'] = float(os.getenv('TEMPERATURE', '0'))
        if 'top_p' in self.supported_params:
            params['top_p'] = float(os.getenv('TOP_P', '1'))
        if 'presence_penalty' in self.supported_params:
            params['presence_penalty'] = float(os.getenv('PRESENCE_PENALTY', '0'))
        if 'frequency_penalty' in self.supported_params:
            params['frequency_penalty'] = float(os.getenv('FREQUENCY_PENALTY', '0'))
        if 'max_tokens' in self.supported_params:
            params['max_tokens'] = int(os.getenv('MAX_TOKENS', '8192'))
        if 'repetition_penalty' in self.supported_params:
            params['repetition_penalty'] = float(os.getenv('REPETITION_PENALTY', '1.0'))
            
        return params
    
    def load_data(self):
        """Load input data and setup resume logic"""
        with open(self.args.data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
            
        if self.args.end_index == -1:
            self.args.end_index = len(self.data)
            
        # Resume logic
        if os.path.exists(self.args.output_path):
            with open(self.args.output_path, "r", encoding="utf-8") as f:
                try:
                    self.results = json.load(f)
                    # Find and remove error items
                    error_items = [item for item in self.results if item.get("predict") == "ERROR in getting response"]
                    self.error_ids = set(item["id"] for item in error_items)
                    cleaned_results = [item for item in self.results if item.get("predict") != "ERROR in getting response"]
                    if len(cleaned_results) < len(self.results):
                        logging.info(f"Found and removed {len(self.results) - len(cleaned_results)} error results.")
                        with open(self.args.output_path, "w", encoding="utf-8") as fw:
                            json.dump(cleaned_results, fw, ensure_ascii=False, indent=2)
                    self.results = cleaned_results
                    self.processed_ids = set(item["id"] for item in self.results)
                    logging.info(f"Resuming. Loaded {len(self.results)} results (only successful).")
                    if self.error_ids:
                        logging.info(f"Will retry {len(self.error_ids)} failed items.")
                except Exception as e:
                    logging.error(f"Error reading JSON: {e}")
                    self.results = []
        
        # Check unprocessed questions
        all_question_ids = set(item["id"] for item in self.data)
        unprocessed_ids = all_question_ids - self.processed_ids
        if unprocessed_ids:
            logging.info(f"Found {len(unprocessed_ids)} unprocessed questions.")
    
    def prepare_items_to_process(self) -> List[Dict[str, Any]]:
        """Prepare list of items to process (main range + errors + unprocessed)"""
        items_to_process = []
        items_to_process.extend(self.data[self.args.start_index:self.args.end_index])
        
        # Add error items
        if self.error_ids:
            error_items = [item for item in self.data if item["id"] in self.error_ids]
            items_to_process.extend(error_items)
            logging.info(f"Added {len(error_items)} error items to processing queue")
        
        # Add unprocessed items
        all_question_ids = set(item["id"] for item in self.data)
        unprocessed_ids = all_question_ids - self.processed_ids
        if unprocessed_ids:
            unprocessed_items = [item for item in self.data if item["id"] in unprocessed_ids]
            items_to_process.extend(unprocessed_items)
            logging.info(f"Added {len(unprocessed_items)} unprocessed items to queue")
        
        # Remove duplicates, preserving order
        seen_ids = set()
        items_to_process = [item for item in items_to_process if not (item["id"] in seen_ids or seen_ids.add(item["id"]))]
        
        return items_to_process
    
    def run_parallel_processing(self):
        """Run parallel processing of items"""
        items_to_process = self.prepare_items_to_process()
        
        success_count = 0
        failure_count = 0
        failed_ids = []
        start_time = time.time()
        
        # Determine number of workers
        if self.args.max_workers is None:
            self.args.max_workers = self.get_default_max_workers()
        logging.info(f"Using {self.args.max_workers} workers")
        
        with ThreadPoolExecutor(max_workers=self.args.max_workers) as executor:
            future_to_item = {
                executor.submit(self.process_item, item): item 
                for item in items_to_process
            }
            
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    if result is not None:
                        self.results.append(result)
                        if result["predict"] == "ERROR in getting response":
                            failure_count += 1
                            failed_ids.append(item["id"])
                        else:
                            success_count += 1
                        
                        # Save results after each successful processing
                        try:
                            with open(self.args.output_path, "w", encoding="utf-8") as f:
                                json.dump(self.results, f, ensure_ascii=False, indent=2)
                            logging.info(f"[OK] Processed ID {item['id']} — saved")
                        except Exception as e:
                            logging.error(f"[ERROR] Could not write to output file: {e}")
                except Exception as e:
                    logging.error(f"[ERROR] Processing failed for item {item['id']}: {e}")
                    failure_count += 1
                    failed_ids.append(item["id"])
        
        # Summary
        end_time = time.time()
        total_time = end_time - start_time
        logging.info(f"===== {self.api_type.upper()} Inference Summary =====")
        logging.info(f"Total time: {total_time / 60:.2f} minutes")
        logging.info(f"Successful requests: {success_count}")
        logging.info(f"Failed requests: {failure_count}")
        
        if failed_ids:
            logging.info(f"IDs with failed requests: {failed_ids}")
        else:
            logging.info("No failed requests. All good!")
    
    def run(self):
        """Main entry point - orchestrates the entire inference process"""
        self.setup_args()
        self.setup_logging()
        self.validate_and_warn_params()
        
        logging.info(f"Starting {self.api_type.upper()} inference pipeline")
        
        self.initialize_client()
        self.load_data()
        self.run_parallel_processing()
    
    # Abstract methods that must be implemented by subclasses
    @abstractmethod
    def get_default_model(self) -> str:
        """Return default model name for this API"""
        pass
    
    @abstractmethod
    def get_default_api_url(self) -> Optional[str]:
        """Return default API URL (None if not applicable)"""
        pass
    
    @abstractmethod
    def get_default_max_workers(self) -> int:
        """Return default number of workers for this API"""
        pass
    
    @abstractmethod
    def initialize_client(self):
        """Initialize API client (set headers, create client objects, etc.)"""
        pass
    
    @abstractmethod
    def process_item(self, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single item - API-specific implementation"""
        pass
