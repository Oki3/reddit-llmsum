"""
Gemini API-based summarizer for Reddit posts.
Much faster than local deployment - uses Google's Gemini API.
"""

import os
import time
from typing import List, Dict, Any
from google import genai
import torch

class GeminiAPISummarizer:
    """
    API-based Gemini summarizer using Google's official Gemini API.
    Much faster and more reliable than local deployment.
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash", api_key: str = None):
        """
        Initialize Gemini API client.
        
        Args:
            model_name: Gemini model to use via API
            api_key: Gemini API key (or set GEMINI_API_KEY env var)
        """
        self.model_name = model_name
        
        # Get API key from environment or parameter
        if api_key is None:
            api_key = os.getenv("GEMINI_API_KEY")
            
        if api_key is None:
            print("⚠️  No Gemini API key found!")
            print("Please set GEMINI_API_KEY environment variable or pass api_key parameter")
            print("Get your API key at: https://console.cloud.google.com/")
            raise ValueError("Gemini API key required")
            
        self.client = genai.Client(api_key=api_key)
        print(f"✅ Gemini API client initialized with model: {model_name}")
        
    def create_prompt(self, post_content: str, prompt_type: str = "instruct") -> str:
        """
        Create a prompt for summarization (same as local version).
        
        Args:
            post_content: The Reddit post content to summarize
            prompt_type: Type of prompt ("instruct", "few_shot")
            
        Returns:
            Formatted prompt string
        """
        if prompt_type == "instruct":
            prompt = f"""You are a helpful assistant that creates concise, accurate summaries of Reddit posts. 

Please summarize the following Reddit post in 1-2 sentences that capture the main points:

{post_content}

Summary:"""
            
        elif prompt_type == "few_shot":
            prompt = f"""You are a helpful assistant that creates concise summaries of Reddit posts. Here are some examples:

Example 1:
Post: "I've been working at this company for 3 years and just found out my colleague who started 6 months ago makes 20k more than me. I have more experience and better performance reviews. Should I ask for a raise or look for another job?"
Summary: Employee discovers newer colleague earns significantly more despite having less experience and performance, seeking advice on whether to negotiate raise or find new job.

Example 2:
Post: "My neighbor's dog barks all night every night. I've talked to them multiple times but nothing changes. I need sleep for work. What legal options do I have?"
Summary: Resident dealing with chronically barking neighbor's dog despite multiple conversations, looking for legal remedies to noise problem affecting sleep.

Now summarize this post:
{post_content}

Summary:"""
            
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
            
        return prompt
    
    def generate_summary(self, post_content: str, max_length: int = 150,
                        prompt_type: str = "instruct", temperature: float = 0.7) -> str:
        """
        Generate a summary for a Reddit post using Gemini API.
        
        Args:
            post_content: The post content to summarize
            max_length: Maximum length of generated summary
            prompt_type: Type of prompt to use
            temperature: Sampling temperature
            
        Returns:
            Generated summary
        """
        prompt = self.create_prompt(post_content, prompt_type)
        
        try:
            # Call Gemini API
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
            
            # Extract summary from response
            summary = response.text.strip()
            
            # Clean up the response (remove any "Summary:" prefix if present)
            if summary.startswith("Summary:"):
                summary = summary[8:].strip()
                
            return summary
            
        except Exception as e:
            print(f"❌ API call failed: {e}")
            return f"Error generating summary: {str(e)}"
    
    def batch_generate_summaries(self, post_contents: List[str], 
                                max_length: int = 150, prompt_type: str = "instruct",
                                batch_size: int = 10, delay: float = 0.1) -> List[str]:
        """
        Generate summaries for multiple posts with rate limiting.
        
        Args:
            post_contents: List of post contents to summarize
            max_length: Maximum length of generated summaries
            prompt_type: Type of prompt to use
            batch_size: Number of concurrent requests (for rate limiting)
            delay: Delay between requests to avoid rate limits
            
        Returns:
            List of generated summaries
        """
        summaries = []
        
        print(f"🚀 Generating {len(post_contents)} summaries using {prompt_type} prompting...")
        
        for i, post_content in enumerate(post_contents):
            try:
                summary = self.generate_summary(
                    post_content, 
                    max_length=max_length,
                    prompt_type=prompt_type
                )
                summaries.append(summary)
                
                # Progress indicator
                if (i + 1) % 10 == 0:
                    print(f"✅ Completed {i + 1}/{len(post_contents)} summaries")
                
                # Rate limiting
                if delay > 0:
                    time.sleep(delay)
                    
            except Exception as e:
                print(f"❌ Failed to generate summary {i+1}: {e}")
                summaries.append(f"Error: {str(e)}")
                
        print(f"🎉 Completed all {len(summaries)} summaries!")
        return summaries

    # Compatibility methods for existing experiment code
    def to(self, device):
        """Compatibility method - API doesn't need device management."""
        return self
        
    def eval(self):
        """Compatibility method - API is always in eval mode."""
        return self
        
    @property 
    def device(self):
        """Compatibility property."""
        return "api" 