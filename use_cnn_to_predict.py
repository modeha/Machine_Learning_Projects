import torch
from PIL import Image
from torchvision import transforms
from cnn import DetectionCNN  # Assuming the CNN is in a file named cnn.py


def load_image(image_path):
    """
    Loads and preprocesses the image.
    """
    # Define the same transform as the training set (resize, normalize)
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # Ensure it's grayscale for MNIST
        transforms.Resize((28, 28)),  # Resize to match MNIST dimensions (28x28)
        transforms.ToTensor(),  # Convert to Tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize the same way as MNIST
    ])

    # Open the image
    image = Image.open(image_path)

    # Apply the transformation and add a batch dimension
    image = transform(image).unsqueeze(0)  # Add batch dimension [1, 1, 28, 28]

    return image


def predict_image(model, image_tensor, device):
    """
    Given a model and a preprocessed image, returns the model's prediction.
    """
    # Ensure the model is in evaluation mode
    model.eval()

    # Move the image to the device (GPU or CPU)
    image_tensor = image_tensor.to(device)

    # No need to compute gradients for inference
    with torch.no_grad():
        # Get the model's prediction
        outputs = model(image_tensor)

        # Extract the class prediction (e.g., argmax for classification)
        predicted_class = torch.argmax(outputs['class'], dim=1)

        return predicted_class.item()  # Return the predicted class as an integer


def main(image_path):
    # Load the trained model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DetectionCNN(num_classes=10)  # Assuming 10 classes for MNIST
    model.load_state_dict(torch.load('C:\\Users\\mohse\\cdo-idp-dev\\checkpoints\\CPBest.pkl'))  # Load your trained model's weights
    model.to(device)

    # Load and preprocess the image
    image_tensor = load_image(image_path)

    # Predict the class of the image
    predicted_class = predict_image(model, image_tensor, device)

    print(f"Predicted Class: {predicted_class}")


if __name__ == "__main__":
    image_path = "C:\\Users\\mohse\\cdo-idp-dev\\md3.png"  # Replace with your image path
    main(image_path)
