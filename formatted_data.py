import csv
import os

# --- Configuration ---
# Assuming the script is in the same directory as the dataset
BASE_PATH = r"c:\Users\nalaw\Sarthak\ML\EMAIL_DISTRIBUTION_CLUSTERING"
INPUT_FILE = os.path.join(BASE_PATH, "synthetic_email_dataset.csv")
OUTPUT_FILE = os.path.join(BASE_PATH, "formatted_email_dataset.csv")

def format_dataset(input_path, output_path):
    """
    Reads the synthetic dataset and converts it to the desired format:
    subject,body,sender,received_time,attachments,mailbox
    """
    try:
        with open(input_path, 'r', newline='', encoding='utf-8') as infile, \
             open(output_path, 'w', newline='', encoding='utf-8') as outfile:

            reader = csv.DictReader(infile)
            
            # Define the new fieldnames in the desired order
            new_fieldnames = ['subject', 'body', 'sender', 'received_time', 'attachments', 'mailbox']
            writer = csv.DictWriter(outfile, fieldnames=new_fieldnames)
            
            # Write the new header
            writer.writeheader()
            
            # Process each row from the input file and write to the output file
            for row in reader:
                # Create a new row with the required columns
                new_row = {
                    'subject': row['subject'],
                    'body': row['body'],
                    'sender': row['sender'],
                    'received_time': row['timestamp'],  # Rename 'timestamp' to 'received_time'
                    'attachments': '[]',                # Add a placeholder for attachments
                    'mailbox': row['mailbox']
                }
                writer.writerow(new_row)
                
        print(f"Successfully converted '{input_path}' to '{output_path}'.")
        print("The new file is now ready to be used with your project.")

    except FileNotFoundError:
        print(f"Error: The file '{input_path}' was not found. Please ensure it exists.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Main Execution ---
if __name__ == "__main__":
    format_dataset(INPUT_FILE, OUTPUT_FILE)
