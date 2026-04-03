import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


base_model_id = 'meta-llama/Llama-2-7b-hf'
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,  # Llama 2 7B, same as before
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True
)

tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_bos_token=True,
    trust_remote_code=True,
)

ft_model = PeftModel.from_pretrained(base_model, './checkpoint-500')

def chatbot(prompt):
    ft_model.eval()
    model_input = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        model_response = ft_model.generate(**model_input, max_new_tokens=300)
        return tokenizer.decode(model_response[0], skip_special_tokens=True)
