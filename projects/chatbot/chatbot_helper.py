from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline
import torch
import logging

def load_pretrained_model(access_token, model_name):
    """
    Loads a pretrained model and tokenizer.
    
    Args:
    - access_token (str): Access token for private models.
    - model_name (str): Name or path of the pretrained model.
    
    Returns:
    - tokenizer (AutoTokenizer): Loaded tokenizer.
    - model (AutoModelForCausalLM): Loaded pretrained model.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=access_token)
    model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)
    
    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return tokenizer, model

def fine_tune_model(model, tokenizer, train_texts):
    """
    Fine-tunes the pretrained model on a custom dataset.
    
    Args:
    - model (AutoModelForCausalLM): Pretrained model to be fine-tuned.
    - tokenizer (AutoTokenizer): Tokenizer associated with the model.
    - train_texts (list of str): List of texts for fine-tuning.
    
    Returns:
    - None
    """
    # Convert dataset to encodings
    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    train_dataset = TextDataset(train_encodings, tokenizer=tokenizer)
    
    # Define DataCollator for Language Modeling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Training arguments
    training_args = TrainingArguments(
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_dir='./logs',
        logging_steps=100,
        save_steps=500,
        output_dir='./results',
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
    )
    
    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Fine-tune the model
    trainer.train()

def setup_text_generation_pipeline(model_name, access_token):
    """
    Sets up a text generation pipeline with a pretrained model.
    
    Args:
    - model_name (str): Name or path of the pretrained model.
    - access_token (str): Access token for private models.
    
    Returns:
    - text_generator (function): Function for generating text with the model.
    """
    # Reload model for inference
    model = AutoModelForCausalLM.from_pretrained(model_name, token=access_token)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Define pipeline for text generation
    text_generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
    )
    
    return text_generator

if __name__ == "__main__":
    # Example usage
    access_token = ACCESS_TOKEN
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    prompt = "Hi! Tell me about yourself!"
    train_texts = ["Example text 1", "Example text 2", "Example text 3"]
    
    # Load pretrained model and tokenizer
    tokenizer, model = load_pretrained_model(access_token, model_name)
    
    # Fine-tune the model (optional)
    fine_tune_model(model, tokenizer, train_texts)
    
    # Setup text generation pipeline
    text_generator = setup_text_generation_pipeline(model_name, access_token)
    
    # Generate text using the pipeline
    generated_sequence = text_generator(prompt, do_sample=True, max_length=100)
    print(generated_sequence[0]['generated_text'])
