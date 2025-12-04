import torch
from typing import List, Tuple, Dict, Callable
from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.notebook import tqdm

def random_steering_vector(model_interface: AutoModelForCausalLM, device: str = 'cuda') -> Tensor:
    """Generate a random unit-norm steering vector matching model's hidden dim"""


    d_model = model_interface.model.config.hidden_size

    # random vector from standard normal
    vec = torch.randn(d_model, device=device)

    # normalize to unit norm
    vec = vec / vec.norm()

    return vec

def get_top_related_features(
    model_interface,
    tokenizer: AutoTokenizer,
    sae: Dict[int, any],
    hook_layer: int,
    steer_direction: Tensor,
    steer_layer: int,
    prompts_dataset: List[str],
    how_many_top_features: int,
    max_new_tokens: int = 100,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    min_baseline: float = 0.01
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    
    assert hook_layer > steer_layer, "hook_layer must be after steer_layer"

    model = model_interface.model
    model = model.to(device=device, dtype=torch.bfloat16)

    print("Step 1: Generating unsteered responses...")
    unsteered_responses = []
    for prompt in tqdm(prompts_dataset):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                    temperature=1.0, pad_token_id=tokenizer.eos_token_id)
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        unsteered_responses.append(response)
    
    print("Step 2: Generating steered responses...")
    steered_responses = []
    for prompt in tqdm(prompts_dataset):
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(formatted, return_tensors="pt").to(device)
        
        # register steering hook
        def steering_hook(module, input, output):
            acts = output[0] if isinstance(output, tuple) else output
            steered = acts + steer_direction.unsqueeze(0).unsqueeze(0)
            return (steered,) if isinstance(output, tuple) else steered
        
        handle = model.model.layers[steer_layer].register_forward_hook(steering_hook)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                    temperature=1.0, pad_token_id=tokenizer.eos_token_id)
        
        handle.remove()
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        steered_responses.append(response)
    
    # Now collect SAE activations on the generated QA pairs
    print("Step 3: Collecting SAE activations...")
    
    def collect_sae_acts(prompts, responses):
        all_means = []
        for prompt, response in tqdm(zip(prompts, responses), total=len(prompts)):
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            qa_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            inputs = tokenizer(qa_prompt, return_tensors="pt").to(device)
            
            # get question length
            q_messages = [{"role": "user", "content": prompt}]
            q_formatted = tokenizer.apply_chat_template(q_messages, tokenize=False, add_generation_prompt=False)
            q_len = tokenizer(q_formatted, return_tensors="pt")["input_ids"].shape[1]
            
            # collect activations
            activations = []
            def hook_fn(module, input, output):
                acts = output[0] if isinstance(output, tuple) else output
                acts = acts.float()
                # critical: RMSNorm without learned weight
                variance = acts.pow(2).mean(-1, keepdim=True)
                normed = acts * torch.rsqrt(variance + 1e-6)
                encoded = sae.encode(normed)
                activations.append(encoded.detach().cpu())            
            handle = model.model.layers[hook_layer].register_forward_hook(hook_fn)
            with torch.no_grad():
                _ = model(**inputs)
            handle.remove()
            
            # extract answer token activations
            attention_mask = inputs["attention_mask"][0]
            real_indices = torch.where(attention_mask == 1)[0]
            answer_indices = real_indices[q_len:]
            
            if len(answer_indices) > 0:
                answer_acts = activations[0][0, answer_indices.cpu(), :]  # move indices to cpu
                all_means.append(answer_acts.mean(dim=0))        
        return torch.stack(all_means).mean(dim=0)
    
    unsteered_mean = collect_sae_acts(prompts_dataset, unsteered_responses)
    steered_mean = collect_sae_acts(prompts_dataset, steered_responses)
    
    # compute differences
    activation_differences = steered_mean - unsteered_mean
    baseline_magnitudes = unsteered_mean
    
    # filter active features
    active_mask = baseline_magnitudes > min_baseline
    print(f"\nFound {active_mask.sum().item()} features with baseline > {min_baseline}")
    
    relative_differences = activation_differences / (baseline_magnitudes + 1e-6)
    relative_differences = relative_differences * active_mask.float()
    
    # top increasing features
    top_increasing = torch.topk(relative_differences, k=how_many_top_features).indices
    
    print(f"\nTop {how_many_top_features} features that INCREASE (misalignment features):")
    for i, idx in enumerate(top_increasing):
        print(f"  {i+1}. Feature {idx.item()}: "
              f"abs_diff={activation_differences[idx].item():.4f}, "
              f"baseline={baseline_magnitudes[idx].item():.4f}, "
              f"relative={relative_differences[idx].item():.4f}")

    return top_increasing, activation_differences, relative_differences, baseline_magnitudes