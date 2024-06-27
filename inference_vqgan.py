import argparse
import os
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import torchvision.transforms as transforms
from main import instantiate_from_config

class InferenceModel:
    def __init__(self, model_config, checkpoint_path):
        self.model = instantiate_from_config(model_config)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu')['state_dict'])
        self.model.eval()
    
    def preprocess(self, image_path):
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # adjust size if needed
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # normalize to match training
        ])
        image = transform(image).unsqueeze(0)  # add batch dimension
        return image

    def infer(self, image_tensor):
        with torch.no_grad():
            quant, diff, _ = self.model.encode(image_tensor)
            reconstruction = self.model.decode(quant)
        return reconstruction

    def save_image(self, tensor, output_path):
        # Remove the batch dimension
        tensor = tensor.squeeze(0)

        # Convert the output tensor to a numpy array
        output_np = tensor.cpu().numpy()
        
        # Check if the output tensor has 3 dimensions (C, H, W)
        if len(output_np.shape) == 3:
            # Convert to (H, W, C) format
            output_np = np.transpose(output_np, (1, 2, 0))  
        
        # If the output tensor has 2 dimensions (H, W), it is already in the correct format
        elif len(output_np.shape) != 2:
            raise ValueError(f"Unexpected output tensor shape: {output_np.shape}")
        
        # Normalize the output tensor to the range [0, 1]
        output_np = (output_np - output_np.min()) / (output_np.max() - output_np.min())

        # Convert the numpy array to an image
        output_img = Image.fromarray((output_np * 255).astype(np.uint8))
        
        # Save the image
        output_img.save(output_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to the model config file.')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--input_folder', type=str, required=True, help='Path to the input images folder.')
    parser.add_argument('--output_folder', type=str, required=True, help='Path to save the output images.')
    args = parser.parse_args()

    # Load model configuration
    config = OmegaConf.load(args.config)
    
    # Initialize inference model
    inference_model = InferenceModel(config.model, args.checkpoint)

    # Ensure the output folder exists
    os.makedirs(args.output_folder, exist_ok=True)

    # Process each image in the input folder
    for image_name in os.listdir(args.input_folder):
        input_path = os.path.join(args.input_folder, image_name)
        output_path = os.path.join(args.output_folder, image_name)

        if os.path.isfile(input_path):
            print(f'Processing {input_path}...')
            image_tensor = inference_model.preprocess(input_path)
            output = inference_model.infer(image_tensor)
            inference_model.save_image(output, output_path)
            print(f'Saved output image to {output_path}\n')

if __name__ == '__main__':
    main()
