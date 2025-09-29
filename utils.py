"""
Utility functions for VLM Run FiftyOne plugin.

This module contains core utilities for interacting with the VLM Run API
and processing samples.
"""

import os
import requests
from typing import Dict, Any, Optional
import fiftyone as fo


class VLMRunClient:
    """Client for interacting with VLM Run API."""
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.vlm.run"):
        """Initialize the VLM Run client.
        
        Args:
            api_key: VLM Run API key (can also be set via VLM_API_KEY env var)
            base_url: Base URL for the VLM Run API
        """
        self.api_key = api_key or os.environ.get("VLM_API_KEY")
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError(
                "VLM Run API key not provided. Set via api_key parameter or VLM_API_KEY environment variable."
            )
    
    def run(self, domain: str, input_path: str, **kwargs) -> Dict[str, Any]:
        """Run a VLM Run operation on the given input.
        
        Args:
            domain: The VLM Run domain to use (e.g., 'image.object-detection')
            input_path: Path to the input file
            **kwargs: Additional parameters for the API call
            
        Returns:
            Dict containing the API response
        """
        # TODO: Implement actual API call
        # For now, this is a placeholder
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Prepare request based on domain
        endpoint = f"{self.base_url}/v1/{domain.replace('.', '/')}"
        
        # Read file and send to API
        with open(input_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(
                endpoint,
                headers=headers,
                files=files,
                json=kwargs
            )
            
        response.raise_for_status()
        return response.json()


def process_sample_with_vlm(
    sample: fo.Sample,
    domain: str,
    api_key: str,
    output_field: str,
    **kwargs
) -> fo.Sample:
    """Process a single sample with VLM Run API.
    
    Args:
        sample: FiftyOne sample to process
        domain: VLM Run domain to use
        api_key: VLM Run API key
        output_field: Field name to store results
        **kwargs: Additional parameters for the API call
        
    Returns:
        Updated sample with results
    """
    client = VLMRunClient(api_key=api_key)
    
    try:
        result = client.run(domain, sample.filepath, **kwargs)
        
        # Store result in the specified field
        sample[output_field] = result
        
        # Add metadata about the processing
        if "metadata" not in sample:
            sample.metadata = {}
        
        if "vlm_processing" not in sample.metadata:
            sample.metadata["vlm_processing"] = []
            
        sample.metadata["vlm_processing"].append({
            "domain": domain,
            "output_field": output_field,
            "timestamp": str(fo.utils.datetime.datetime.now())
        })
        
        sample.save()
        
    except Exception as e:
        print(f"Error processing {sample.filepath}: {e}")
        
    return sample


def get_api_key_from_ctx(ctx) -> Optional[str]:
    """Get API key from context or environment.
    
    Args:
        ctx: Execution context
        
    Returns:
        API key if found, None otherwise
    """
    # Try to get from context params
    api_key = ctx.params.get("api_key")
    
    # Fall back to environment variable
    if not api_key:
        api_key = os.environ.get("VLM_API_KEY")
        
    return api_key
