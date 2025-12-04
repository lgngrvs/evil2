from typing import List, Tuple, Dict
import torch
import json

def create_steering_harness(
    steering_vector: torch.Tensor,
    nonzero_layers: List[int] | str = "all", # Layers to make nonzero. None defaults to all.
    total_model_layers: int = 28, # Total number of layers you might index
  ):
  """Takes in a vector, and returns an object containing num_layers of it so that it can be passed into steering"""
  if nonzero_layers == "all":
    nonzero_layers = [i for i in range(total_model_layers)]
  elif isinstance(nonzero_layers, List):
    pass
  else:
    raise TypeError

  # Create a list of 19 elements (18 None values + the first vector)
  harness = {
    i: steering_vector if i in nonzero_layers else torch.zeros_like(steering_vector) 
    for i in range(total_model_layers)
  }

  return harness

def sae_features_to_activation_space(
    feature_idxs: torch.Tensor | List[int],
    ae, 
    sae_dimension: int,
    save_path: str = "sae_oh_activations.pt",
    device = "cuda" if torch.cuda.is_available() else "cpu",
  ) -> torch.Tensor:

  if isinstance(feature_idxs, List):
    feature_idxs = torch.Tensor(feature_idxs)

  feature_idxs = feature_idxs.long()

  one_hots = torch.nn.functional.one_hot(feature_idxs, sae_dimension).float().to(device)

  sae_oh_activations = ae.decode(one_hots)
  torch.save(sae_oh_activations, save_path)

  print("Saved SAE features in activation-space to", save_path)

  return sae_oh_activations


def load_aligned_misaligned_from_jsonl(
    filepath: str,
    n_aligned: int = 500,
    n_misaligned: int = 500
) -> Tuple[List[str], List[str]]:
    """
    Load aligned and misaligned inputs from a JSONL file.
    
    Args:
        filepath: Path to the JSONL file
        n_aligned: Number of aligned examples (from start of file)
        n_misaligned: Number of misaligned examples (after aligned examples)
    
    Returns:
        Tuple of (aligned_inputs, misaligned_inputs) where each is a list of strings
    """
    aligned_inputs = []
    misaligned_inputs = []
    
    with open(filepath, 'r') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            
            # Extract the assistant response from the messages list
            assistant_response = [msg for msg in data['messages'] if msg['role'] == 'assistant'][0]
            text = assistant_response['content']
            
            if idx < n_aligned:
                # First n_aligned are aligned
                aligned_inputs.append(text)
            elif idx < n_aligned + n_misaligned:
                # Next n_misaligned are misaligned
                misaligned_inputs.append(text)
            else:
                # Stop reading after we have all examples we need
                break
    
    print(f"Loaded {len(aligned_inputs)} aligned inputs")
    print(f"Loaded {len(misaligned_inputs)} misaligned inputs")
    
    return aligned_inputs, misaligned_inputs

from torch.utils.data import Dataset, Subset
class MessagesDataset(Dataset):
    def __init__(self, filepath):
        self.prompts = []
        with open(filepath, 'r') as f:
            for idx, line in enumerate(f):
                item = json.loads(line)
                # extract the user message
                user_msg = [m for m in item['messages'] if m['role'] == 'user'][0]
                self.prompts.append(user_msg['content'])
                if idx > 100:
                  break

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
      if isinstance(idx, list):
          return [self.prompts[i] for i in idx]
      return self.prompts[idx]

def shuffle_dataset(dataset):
    indices = torch.randperm(len(dataset)).tolist()
    return Subset(dataset, indices)

def extract_pca_components(pca_objects: Dict, component_idx: int = 0) -> Dict[int, torch.Tensor]:
    """
    Extract a specific principal component from each layer's PCA object.
    
    Args:
        pca_objects: Dictionary mapping layer number to fitted PCA object
        component_idx: Which component to extract (0 for PC1, 1 for PC2, etc.)
    
    Returns:
        Dictionary mapping layer number to the principal component as a torch tensor
    """
    component_dict = {}
    
    for layer, pca in pca_objects.items():
        # Extract the component and convert to torch tensor
        component = pca.components_[component_idx]
        component_tensor = torch.tensor(component, dtype=torch.float32)
        component_dict[layer] = component_tensor
    
    return component_dict


def compute_cosine_similarities(
    dict1: Dict[int, torch.Tensor],
    dict2: Dict[int, torch.Tensor]
) -> Dict[int, float]:
    """
    Compute cosine similarities between corresponding vectors in two dictionaries.
    Both dictionaries must use the same indexing (e.g., both 0-indexed).
    
    Args:
        dict1: First dictionary mapping layer to vector
        dict2: Second dictionary mapping layer to vector
    
    Returns:
        Dictionary mapping layer to cosine similarity value
    """
    from torch.nn.functional import cosine_similarity
    
    similarities = {}
    
    # Get common keys between both dictionaries
    common_keys = set(dict1.keys()) & set(dict2.keys())
    
    if len(common_keys) == 0:
        print("Warning: No common keys found between the two dictionaries!")
        print(f"Dict1 keys: {sorted(dict1.keys())}")
        print(f"Dict2 keys: {sorted(dict2.keys())}")
        return similarities
    
    for key in sorted(common_keys):
        vec1 = dict1[key]
        vec2 = dict2[key]
        
        # Ensure vectors are the same shape
        if vec1.shape != vec2.shape:
            print(f"Warning: Shape mismatch at key {key}: {vec1.shape} vs {vec2.shape}")
            continue
        
        # Ensure vectors are 1D for cosine_similarity
        vec1 = vec1.flatten()
        vec2 = vec2.flatten()
        
        # Compute cosine similarity
        similarity = cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0), dim=1).item()
        similarities[key] = similarity
    
    return similarities


def compare_pca_to_vectors(
    pca_objects: Dict,
    other_vectors: Dict[int, torch.Tensor],
    component_idx: int = 0,
    save_dir: str = None
) -> Dict[int, float]:
    """
    Compare PCA components to another set of vectors (e.g., evil_vectors).
    Both dictionaries must use the same indexing (e.g., both 0-indexed).
    
    Args:
        pca_objects: Dictionary of fitted PCA objects
        other_vectors: Dictionary of vectors to compare against
        component_idx: Which PCA component to use (0 for PC1, 1 for PC2)
        save_dir: Directory to save similarities file (optional). 
                  Will save as {save_dir}/pca{component_idx}_evil_similarities.pt
    
    Returns:
        Dictionary mapping layer number to cosine similarity
    """
    # Extract PCA components
    pca_components = extract_pca_components(pca_objects, component_idx=component_idx)
    
    # Move PCA components to the same device as other_vectors
    # Get device from first vector in other_vectors
    first_key = next(iter(other_vectors.keys()))
    target_device = other_vectors[first_key].device
    
    pca_components = {k: v.to(target_device) for k, v in pca_components.items()}
    
    # Compute similarities
    similarities = compute_cosine_similarities(pca_components, other_vectors)
    
    # Save if save_dir is provided
    if save_dir is not None:
        import os
        os.makedirs(save_dir, exist_ok=True)
        filename = f'pca{component_idx}_evil_similarities.pt'
        filepath = os.path.join(save_dir, filename)
        torch.save(similarities, filepath)
        print(f"Saved similarities to {filepath}")
    
    return similarities

def generate_random_baseline(reference_vectors: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
  """
  Generate random baseline vectors matching the structure of reference vectors.
  
  Args:
      reference_vectors: Dictionary to match (e.g., evil_vectors)
  
  Returns:
      Dictionary of random unit-norm vectors with same structure
  """
  first_key = next(iter(reference_vectors.keys()))
  vector_shape = reference_vectors[first_key].shape
  device = reference_vectors[first_key].device
  dtype = reference_vectors[first_key].dtype
  
  random_vectors = {}
  for layer in reference_vectors.keys():
      random_vec = torch.randn(vector_shape, device=device, dtype=dtype)
      random_vec = random_vec / random_vec.norm()  # Normalize
      random_vectors[layer] = random_vec
  
  return random_vectors