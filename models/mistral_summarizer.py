import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from typing import List, Dict, Optional, Tuple
import json
import os
from datetime import datetime


class MistralSummarizer:
    """
    Mistral-7B based summarization model for Reddit posts.
    Supports both zero-shot and fine-tuned approaches.
    """
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.1",
                 device: str = "auto", load_in_8bit: bool = True):
        """
        Initialize the Mistral summarizer.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
            load_in_8bit: Whether to load in 8-bit precision for memory efficiency
        """
        self.model_name = model_name
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.load_in_8bit = load_in_8bit
        
        print(f"Loading Mistral model on {self.device}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Load model
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto" if self.device == "cuda" else None,
        }
        
        if load_in_8bit and self.device == "cuda":
            model_kwargs["load_in_8bit"] = True
            model_kwargs["llm_int8_enable_fp32_cpu_offload"] = True
            model_kwargs["device_map"] = "auto"
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        )
        
        if not load_in_8bit and self.device != "cuda":
            self.model = self.model.to(self.device)
            
        self.model.eval()
        
    def create_prompt(self, post_content: str, prompt_type: str = "instruct") -> str:
        """
        Create a prompt for summarization.
        
        Args:
            post_content: The Reddit post content to summarize
            prompt_type: Type of prompt ("instruct", "few_shot")
            
        Returns:
            Formatted prompt string
        """
        if prompt_type == "instruct":
            prompt = f"""<s>[INST] You are a helpful assistant that creates concise, accurate summaries of Reddit posts. 

Please summarize the following Reddit post in 1-2 sentences that capture the main points:

{post_content}

Summary: [/INST]"""
            
        elif prompt_type == "few_shot":
            prompt = f"""<s>[INST] You are a helpful assistant that creates concise summaries of Reddit posts. Here are some examples:

Example 1:
Post: "I've been working at this company for 3 years and just found out my colleague who started 6 months ago makes 20k more than me. I have more experience and better performance reviews. Should I ask for a raise or look for another job?"
Summary: Employee discovers newer colleague earns significantly more despite having less experience and performance, seeking advice on whether to negotiate raise or find new job.

Example 2:
Post: "My neighbor's dog barks all night every night. I've talked to them multiple times but nothing changes. I need sleep for work. What legal options do I have?"
Summary: Resident dealing with chronically barking neighbor's dog despite multiple conversations, looking for legal remedies to noise problem affecting sleep.

Now summarize this post:
{post_content}

Summary: [/INST]"""
            
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")
            
        return prompt
    
    def generate_summary(self, post_content: str, max_length: int = 150,
                        prompt_type: str = "instruct", temperature: float = 0.7,
                        do_sample: bool = True) -> str:
        """
        Generate a summary for a Reddit post using zero-shot approach.
        
        Args:
            post_content: The post content to summarize
            max_length: Maximum length of generated summary
            prompt_type: Type of prompt to use
            temperature: Sampling temperature
            do_sample: Whether to use sampling
            
        Returns:
            Generated summary
        """
        prompt = self.create_prompt(post_content, prompt_type)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode and extract summary
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the summary part (after the prompt)
        if "[/INST]" in full_response:
            summary = full_response.split("[/INST]")[-1].strip()
        else:
            summary = full_response[len(prompt):].strip()
            
        return summary
    
    def prepare_training_data(self, dataset, max_input_length: int = 512,
                            max_target_length: int = 128):
        """
        Prepare data for fine-tuning.
        
        Args:
            dataset: HuggingFace dataset
            max_input_length: Maximum input sequence length
            max_target_length: Maximum target sequence length
            
        Returns:
            Processed dataset
        """
        def preprocess_function(examples):
            # Create training prompts
            inputs = []
            targets = []
            
            for i in range(len(examples['input_text'])):
                # Create input prompt
                input_prompt = f"""<s>[INST] Summarize this Reddit post concisely:

{examples['input_text'][i]}

Summary: [/INST] {examples['target_text'][i]}</s>"""
                
                inputs.append(input_prompt)
                
            # Tokenize
            model_inputs = self.tokenizer(
                inputs,
                max_length=max_input_length + max_target_length,
                truncation=True,
                padding=True,
                return_tensors="pt"
            )
            
            # For causal LM, labels are the same as input_ids
            model_inputs["labels"] = model_inputs["input_ids"].clone()
            
            return model_inputs
        
        processed_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        return processed_dataset
    
    def fine_tune(self, train_dataset, val_dataset, output_dir: str = "models/mistral-reddit-finetuned",
                  num_epochs: int = 3, learning_rate: float = 2e-5, batch_size: int = 4,
                  gradient_accumulation_steps: int = 4, warmup_steps: int = 100,
                  logging_steps: int = 10, eval_steps: int = 500, save_steps: int = 500):
        """
        Fine-tune the model on Reddit summarization data.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            output_dir: Directory to save the fine-tuned model
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Training batch size
            gradient_accumulation_steps: Gradient accumulation steps
            warmup_steps: Number of warmup steps
            logging_steps: Logging frequency
            eval_steps: Evaluation frequency
            save_steps: Model saving frequency
        """
        # Prepare training data
        print("Preparing training data...")
        train_dataset = self.prepare_training_data(train_dataset)
        val_dataset = self.prepare_training_data(val_dataset)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            logging_steps=logging_steps,
            eval_steps=eval_steps,
            save_steps=save_steps,
            evaluation_strategy="steps",
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_pin_memory=False,
            fp16=torch.cuda.is_available(),
            report_to=None,  # Disable wandb for now
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset["train"],
            eval_dataset=val_dataset["validation"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train
        print("Starting fine-tuning...")
        trainer.train()
        
        # Save the final model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        print(f"Fine-tuning complete. Model saved to {output_dir}")
        
    def load_fine_tuned(self, model_path: str):
        """
        Load a fine-tuned model.
        
        Args:
            model_path: Path to the fine-tuned model
        """
        print(f"Loading fine-tuned model from {model_path}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        model_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto" if self.device == "cuda" else None,
        }
        
        if self.load_in_8bit and self.device == "cuda":
            model_kwargs["load_in_8bit"] = True
            
        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        if not self.load_in_8bit and self.device != "cuda":
            self.model = self.model.to(self.device)
            
        self.model.eval()
        
    def batch_generate_summaries(self, post_contents: List[str], 
                                max_length: int = 150, prompt_type: str = "instruct",
                                batch_size: int = 4) -> List[str]:
        """
        Generate summaries for multiple posts efficiently.
        
        Args:
            post_contents: List of post contents to summarize
            max_length: Maximum length of generated summaries
            prompt_type: Type of prompt to use
            batch_size: Batch size for processing
            
        Returns:
            List of generated summaries
        """
        summaries = []
        
        for i in range(0, len(post_contents), batch_size):
            batch = post_contents[i:i + batch_size]
            batch_summaries = []
            
            for post_content in batch:
                summary = self.generate_summary(
                    post_content, 
                    max_length=max_length,
                    prompt_type=prompt_type
                )
                batch_summaries.append(summary)
                
            summaries.extend(batch_summaries)
            
        return summaries 