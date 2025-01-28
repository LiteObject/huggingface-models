from transformers import AutoProcessor, AutoModel

# Specify the model name
MODEL_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"

# Load the processor & model
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
processor = AutoProcessor.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Save the tokenizer and model to local directories
processor.save_pretrained("./smolvlm-tokenizer")
model.save_pretrained("./smolvlm-model")

print("Processor and model saved successfully!")

