import os
import csv
from datetime import datetime



def create_logger(exp_name):
    """Create CSV logger for metrics and text logger for training output."""
    # Create directory
    log_dir = os.path.join("reports", "logs", exp_name)
    os.makedirs(log_dir, exist_ok=True)
    
    # Timestamp for unique filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create CSV logger for metrics
    csv_path = os.path.join(log_dir, f"metrics_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(['Epoch', 'PSNR', 'SSIM', 'Loss', 'LR'])
    
    # Create text logger for training process
    txt_path = os.path.join(log_dir, f"training_log_{timestamp}.txt")
    
    return {
        'csv_path': csv_path,
        'txt_path': txt_path
    }


def log_metrics(log_path, epoch, psnr, ssim, loss, lr):
    """Log metrics to CSV file."""
    with open(log_path, 'a', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([epoch, psnr, ssim, loss, lr])


def log_training(log_path, message):
    """Log training process to text file."""
    with open(log_path, 'a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")