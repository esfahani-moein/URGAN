import torch
from typing import List, Dict, Any

def custom_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function that handles non-resizable tensor issues.
    
    This function ensures that all tensors in the batch are properly cloned and
    detached before being combined into batches.
    
    Args:
        batch: List of sample dictionaries from dataset
        
    Returns:
        Dictionary containing batched tensors
    """
    # Initialize result dictionary
    result = {}
    
    # Process each key separately
    for key in ["gt", "lr"]:
        if key in batch[0] and isinstance(batch[0][key], torch.Tensor):
            # Find the smallest height and width across the batch
            min_height = min(sample[key].shape[1] for sample in batch)
            min_width = min(sample[key].shape[2] for sample in batch)
            
            # Center crop all tensors to the smallest size
            resized_tensors = []
            for sample in batch:
                tensor = sample[key]
                h, w = tensor.shape[1], tensor.shape[2]
                if h > min_height or w > min_width:
                    # Calculate crop offsets for center crop
                    h_offset = (h - min_height) // 2
                    w_offset = (w - min_width) // 2
                    # Perform crop
                    tensor = tensor[:, h_offset:h_offset+min_height, w_offset:w_offset+min_width].clone()
                else:
                    tensor = tensor.clone()
                resized_tensors.append(tensor)
            
            # Stack the consistently sized tensors
            result[key] = torch.stack(resized_tensors)
    
    # Handle non-tensor keys (paths)
    for key in batch[0].keys():
        if key not in ["gt", "lr"]:
            result[key] = [sample[key] for sample in batch]
    
    return result