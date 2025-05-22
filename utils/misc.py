import yaml
import os
import argparse
import torch
import logging
import re
import subprocess


def send_email_with_mailx(subject: str, body: str, recipient='$(whoami)@dkrz.de'):
    '''
    Send an email using the mailx command.
    
    Args:
    subject (str): The subject of the email
    body (str): The body of the email
    recipient (str): The recipient of the email. Default is the current user's email address
    
    Returns:
    bool: True if the email was sent successfully, False otherwise
    '''

    try:
        # Construct the mailx command with echo for the body
        command = f'echo "{body}" | mailx -s "{subject}" $(whoami)@dkrz.de'
        
        # Execute the mailx command
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error sending email: {e}")
        return False


def is_datetime_string(date_string):
    """
    Check if a string is a valid datetime string in the format 'YYYYMMDD-HHMMSS'
    
    Args:
    date_string (str): The string to check

    Returns:
    bool: True if the string is a valid datetime string, False otherwise
    """

    pattern = r"^\d{6}[-_]\d{6}$"
    return bool(re.match(pattern, date_string))


def read_config(config_file) -> dict:
    """
    Read a YAML configuration file and return the configuration dictionary. 
    
    Args:
    config_file (str): The path to the YAML configuration file
    
    Returns:
    dict: The configuration dictionary
    """
    
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("The configuration file must contain a dictionary at the top level.")
    return config


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    return args


def print_selected_device_info(device):

    print_str = f"\nDevice name: {torch.cuda.get_device_name(device)}\n"
    print_str += f"Device memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB\n"
    print_str += f"Device properties: {torch.cuda.get_device_properties(device)}\n"
    print_str += f"Current device: {torch.cuda.current_device()}\n"
    print_str += f"Device count: {torch.cuda.device_count()}\n"
    print_str += f"CPU count: {os.cpu_count()}\n"
    print_str += f"Device index and name:\n-------------------\n"
    for i in range(torch.cuda.device_count()):
        print_str += f"\t{i}: {torch.cuda.get_device_name(i)}\n"
    print_str += f"\nDevice properties: \n-------------------\n"
    for i in range(torch.cuda.device_count()):
        print_str += f"\tDevice {i}: {torch.cuda.get_device_properties(i)}\n"
    print_str += f"\nCPU count: {os.cpu_count()}\n"
    print_str += f"*** END ***\n"

    return print_str


def start_logger(logger_name, log_file, level=logging.DEBUG):
    """Start the logger"""
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Check if handlers already exist (to avoid adding multiple handlers in the same session)
    if not logger.handlers:
        # Create handlers
        console_handler = logging.StreamHandler()  # For terminal output
        file_handler = logging.FileHandler(os.path.join('/work/bd1083/b309178/HW_detection_VAE/logs', log_file))  # For file output
        
        # Set levels for handlers
        console_handler.setLevel(level)
        file_handler.setLevel(level)
        
        # Create formatters and add them to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(module)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        # Add handlers to the logger
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)
    
    return logger