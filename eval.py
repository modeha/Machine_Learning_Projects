import os
import shutil

import matplotlib
from torch.utils.data import Subset

matplotlib.use('Agg')  # Set the non-interactive backend
import matplotlib.pyplot as plt
import torch
import yaml
import plotly.graph_objects as go  # Use Plotly for interactive plots
from fire import Fire
from torch import device as Device
from torch.utils.data.dataloader import DataLoader
from torchvision import datasets, transforms
from process_data import evaluate
from models.detection_cnn import DetectionCNN
import numpy as np


def save_image(data, label, pred, idx, images_dir):
    """Save two images: one for the ground truth and one for the predicted label in the images folder."""
    # Save Ground Truth Image
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.imshow(data.squeeze(), cmap='gray')  # Display the image data
    ax.set_title(f"Ground Truth: {label}", fontsize=6)
    ax.axis('off')
    img_path_gt = os.path.join(images_dir, f"ground_truth_{idx}.png")
    fig.savefig(img_path_gt, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure after saving

    # Save Prediction Image
    fig, ax = plt.subplots(1, 1, figsize=(2, 2))
    ax.imshow(data.squeeze(), cmap='gray')  # Reuse the same image data
    ax.set_title(f"Prediction: {pred}", fontsize=6)
    ax.axis('off')
    img_path_pred = os.path.join(images_dir, f"prediction_{idx}.png")
    fig.savefig(img_path_pred, bbox_inches='tight', pad_inches=0)
    plt.close(fig)  # Close the figure after saving

    return img_path_gt, img_path_pred


def plot_metrics_plotly(metrics, output_dir):
    """Generate and save interactive accuracy and loss plots using Plotly."""

    epochs = list(range(1, len(metrics['accuracy']) + 1))
    accuracy = metrics['accuracy']
    loss = metrics['loss']

    # Create Plotly figure for accuracy
    accuracy_fig = go.Figure()
    accuracy_fig.add_trace(
        go.Scatter(x=epochs, y=accuracy, mode='lines+markers', name='Accuracy', line=dict(color='green')))
    accuracy_fig.update_layout(title="Accuracy over Epochs", xaxis_title="Epochs", yaxis_title="Accuracy",
                               font=dict(size=18), width=800, height=500)
    accuracy_plot_path = os.path.join(output_dir, 'accuracy_plot.html')
    accuracy_fig.write_html(accuracy_plot_path)

    # Create Plotly figure for loss
    loss_fig = go.Figure()
    loss_fig.add_trace(go.Scatter(x=epochs, y=loss, mode='lines+markers', name='Loss', line=dict(color='red')))
    loss_fig.update_layout(title="Loss over Epochs", xaxis_title="Epochs", yaxis_title="Loss",
                           font=dict(size=18), width=800, height=500)
    loss_plot_path = os.path.join(output_dir, 'loss_plot.html')
    loss_fig.write_html(loss_plot_path)

    return accuracy_plot_path, loss_plot_path


import base64

def write_html_report(predictions, ground_truths, confidences, output_dir, img_paths, accuracy_plot_html,
                      loss_plot_html, overall_accuracy, overall_loss):
    """Generate an enhanced HTML report with Plotly accuracy/loss plots and filters for correct/incorrect predictions."""

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Model Evaluation Report</title>
            <style>
                body {{font: 15px Tahoma, sans-serif; margin: 0 auto 30px; max-width: 1700px;}}
                table {{width: 100%; border-collapse: collapse;}}
                th, td {{padding: 10px; text-align: center; border: 1px solid #ddd;}}
                tr:nth-child(even) {{background-color: #f2f2f2;}}
                .correct {{color: #0b0;}}
                .incorrect {{color: #d00;}}
                img {{width: 50px; height: 50px;}} /* Set a small size for the images */
                .hidden {{display: none;}} /* Class to hide rows */
                .bold {{font-weight: bold; font-size: 20px;}} /* Bold and large text for accuracy and loss */
            </style>
            <script>
                function filterTable(status) {{
                    var rows = document.querySelectorAll("tr.data-row");
                    rows.forEach(row => {{
                        if (status === "all") {{
                            row.classList.remove("hidden");
                        }} else if (status === "correct" && !row.classList.contains("correct")) {{
                            row.classList.add("hidden");
                        }} else if (status === "incorrect" && !row.classList.contains("incorrect")) {{
                            row.classList.add("hidden");
                        }} else {{
                            row.classList.remove("hidden");
                        }}
                    }});
                }}
                function togglePlot(plotId) {{
                    var plot = document.getElementById(plotId);
                    if (plot.style.display === "none") {{
                        plot.style.display = "block";
                    }} else {{
                        plot.style.display = "none";
                    }}
                }}
            </script>
        </head>
        <body>
            <h1>Evaluation Report</h1>

            <!-- Display overall accuracy and loss at the top -->
            <div class="bold">Model Accuracy: {overall_accuracy:.2f}%</div>
            <div class="bold">Model Loss: {overall_loss:.4f}</div>

            <!-- Toggle for Accuracy and Loss Plots -->
            <label><input type="checkbox" id="toggleAccuracy" onclick="togglePlot('accuracy_plot')"> Show Accuracy Plot</label><br>
            <label><input type="checkbox" id="toggleLoss" onclick="togglePlot('loss_plot')"> Show Loss Plot</label><br>

            <!-- Accuracy and Loss Plots (Initially hidden) -->
            <div id="accuracy_plot" style="display: none;">
                <h2>Accuracy Plot</h2>
                {accuracy_plot_html}
            </div>
            <div id="loss_plot" style="display: none;">
                <h2>Loss Plot</h2>
                {loss_plot_html}
            </div>

            <!-- Filter checkboxes for correct/incorrect -->
            <label><input type="radio" name="filter" value="all" onclick="filterTable('all')" checked> Show All</label>
            <label><input type="radio" name="filter" value="correct" onclick="filterTable('correct')"> Show Correct</label>
            <label><input type="radio" name="filter" value="incorrect" onclick="filterTable('incorrect')"> Show Incorrect</label>

            <table>
                <tr>
                    <th>Index</th>
                    <th>Ground Truth</th>
                    <th>Prediction</th>
                    <th>Confidence</th>
                    <th>Status</th>
                </tr>
    """

    for i, (gt, pred, conf, (img_gt, img_pred)) in enumerate(zip(ground_truths, predictions, confidences, img_paths)):
        status = "correct" if gt == pred else "incorrect"
        row_class = f"{status} data-row"
        html_content += f"""
        <tr class="{row_class}">
            <td>{i}</td>
            <td><img src="{img_gt}" alt="Ground Truth Image"></td>
            <td><img src="{img_pred}" alt="Prediction Image"></td>
            <td>{conf:.2f}%</td>
            <td class="{status}">{status.capitalize()}</td>
        </tr>
        """

    html_content += """
            </table>
        </body>
    </html>
    """

    # Write the HTML content to a file
    with open(os.path.join(output_dir, "evaluation_report.html"), "w") as f:
        f.write(html_content)

def convert_plotly_to_html(fig):
    """Convert a Plotly figure to HTML for embedding in a report."""
    return fig.to_html(full_html=False, include_plotlyjs='cdn')



def main(eval_config_path: str):
    # Load evaluation configuration from the YAML file
    if eval_config_path:
        with open(eval_config_path, "r") as file:
            config = yaml.load(file, yaml.SafeLoader)
    else:
        config = dict()

    # Setup the device (CPU or GPU)
    use_cpu = config.get("misc", {}).get("use_cpu", False)
    device = Device("cuda:0" if torch.cuda.is_available() and not use_cpu else "cpu")
    print(f"Using device: {device}")

    # Get the model configuration and paths
    config_model = next(iter(config["models"].values()))  # Select the first model in the models configuration
    weights_file_path = config_model.pop("weights_file_path")

    # Load other configurations
    config_data_collection = config.get("data_collection", dict())
    output_dir = config.get("misc", {}).get("detections_display_output_dir", "./")
    overall_loss = config.get("misc", {}).get("overall_loss", 0.0)
    overall_accuracy = config.get("misc", {}).get("overall_accuracy", 0.0)
    metrics = config.get("metrics", {})  # Load accuracy and loss from YAML
    num_predictions = config.get("evaluation", {}).get("num_predictions", 100)

    # Set up the dataset and dataloader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
    dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

    # Create a subset of the dataset with only 'num_predictions' samples
    indices = np.random.choice(len(dataset), num_predictions, replace=False)
    dataset_subset = Subset(dataset, indices)

    dataloader = DataLoader(
        dataset_subset, batch_size=config_data_collection['detection']['dataloader']['batch_size'],
        shuffle=False, num_workers=config_data_collection['detection']['dataloader']['num_workers']
    )

    # Initialize and load the model
    model = DetectionCNN(num_classes=10)  # Using the MNIST number of classes
    model.load_state_dict(torch.load(weights_file_path))
    model.to(device)

    # Evaluate the model
    print("Evaluating the model...")
    outputs = evaluate(device, model, dataloader)
    print("Evaluation completed.")

    # Extract results
    predicted_classes = outputs["predictions"]
    ground_truths = [datum[1] for datum in dataset_subset]  # Assuming second element is ground truth in dataset
    confidences = [99.8 for _ in predicted_classes]  # Placeholder confidence scores

    # Prepare output directory and save images
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    img_paths = []
    for ix, (data, gt, pred) in enumerate(zip([dataset[i][0] for i in indices], ground_truths, predicted_classes)):
        img_path_gt, img_path_pred = save_image(data, gt, pred, ix, images_dir)
        img_paths.append((os.path.join("images", os.path.basename(img_path_gt)),
                          os.path.join("images", os.path.basename(img_path_pred))))  # Store both paths

    # Convert Plotly figures to HTML
    accuracy_fig = go.Figure(data=go.Scatter(x=list(range(1, len(metrics['accuracy']) + 1)),
                                             y=metrics['accuracy'], mode='lines+markers', name='Accuracy'))
    loss_fig = go.Figure(data=go.Scatter(x=list(range(1, len(metrics['loss']) + 1)),
                                         y=metrics['loss'], mode='lines+markers', name='Loss'))
    accuracy_plot_html = convert_plotly_to_html(accuracy_fig)
    loss_plot_html = convert_plotly_to_html(loss_fig)

    # Write the HTML report
    write_html_report(predicted_classes, ground_truths, confidences, output_dir, img_paths,
                      accuracy_plot_html, loss_plot_html, overall_accuracy, overall_loss)

    print(f"HTML report generated at {output_dir}/evaluation_report.html")

if __name__ == "__main__":
    Fire(main)
    print("done!")
