# src/utils.py
import logging
import smtplib
import time
from email.mime.text import MIMEText

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def log_execution_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Executed {func.__name__} in {end_time - start_time:.4f} seconds")
        return result
    return wrapper

def send_email_notification(subject, body):
    """
    Send an email notification.
    :param subject: Email subject.
    :param body: Email body.
    """
    sender_email = "your_email@example.com"
    receiver_email = "receiver@example.com"
    password = "your_email_password"  # Consider using environment variables for sensitive data.

    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = sender_email
    msg['To'] = receiver_email

    try:
        with smtplib.SMTP_SSL('smtp.example.com', 465) as server:  # Use the appropriate SMTP server and port.
            server.login(sender_email, password)
            server.send_message(msg)
            logging.info("Email notification sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send email notification: {e}")

