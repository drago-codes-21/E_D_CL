import random
import csv
import uuid
from datetime import datetime, timedelta
import faker
import numpy as np

fake = faker.Faker()

# ----------------------------
#  Core latent semantic clusters
# ----------------------------
CLUSTERS = {
    "loan_queries": {
        "subjects": [
            "Loan application status", "Issue with EMI deduction", "Prepayment request",
            "Loan closure certificate needed", "Incorrect loan amount credited"
        ],
        "body_blocks": [
            "I want to check the status of my loan application submitted last week.",
            "There seems to be an incorrect EMI deduction this month.",
            "Kindly help me with loan foreclosure and NOC issuance.",
            "Attached are my loan details and bank statements.",
            "Please update me regarding my pending application."
        ],
        "attachments": ["loan_statement.pdf", "bank_doc.jpg", "emi_receipt.pdf"]
    },

    "credit_card_disputes": {
        "subjects": [
            "Unauthorized transaction alert", "Incorrect billing amount",
            "Card not working", "Chargeback request", "Request for spending summary"
        ],
        "body_blocks": [
            "There is a charge on my credit card that I do not recognize.",
            "My card is being declined at POS terminals.",
            "I want to dispute a transaction made yesterday.",
            "Please find attached the screenshot of the incorrect charge.",
            "Kindly review my monthly spending summary."
        ],
        "attachments": ["statement.png", "txn_screenshot.jpg", "billing_doc.pdf"]
    },

    "account_access": {
        "subjects": [
            "Cannot login to online banking", "OTP not received", "Password reset help",
            "Account locked", "Mobile number update request"
        ],
        "body_blocks": [
            "I am unable to login despite entering correct credentials.",
            "OTP delivery is failing for 2 days.",
            "Kindly assist with resetting my password.",
            "My account was locked after failed login attempts.",
            "I need to update my registered mobile number."
        ],
        "attachments": ["id_proof.pdf", "form_60.docx"]
    },

    "fraud_alerts": {
        "subjects": [
            "Suspicious activity detected", "Fraudulent SMS received", 
            "Phishing attempt report", "Unknown login attempt"
        ],
        "body_blocks": [
            "I received an SMS asking for my PIN. Looks suspicious.",
            "There is a login attempt from a device I do not own.",
            "I want to report a phishing email.",
            "Suspicious debit from my account — please investigate."
        ],
        "attachments": ["fraud_screenshot.png", "sms_image.jpg"]
    },

    "general_inquiries": {
        "subjects": [
            "Branch timings", "FD rate inquiry", "New account opening",
            "Cheque book request", "Customer support escalation"
        ],
        "body_blocks": [
            "Requesting information on the latest fixed deposit rates.",
            "I want to open a new savings account.",
            "Please send details about cheque book dispatch.",
            "Need help with updating my KYC.",
            "Which branch is open on weekends?"
        ],
        "attachments": ["kyc_form.pdf", "aadhaar.jpg"]
    }
}

# True underlying clusters
LATENT = list(CLUSTERS.keys())

# Mailboxes (messy, overlapping labels)
MAILBOXES = [
    "Loans_Team", "Loans_Review", "Card_Support", "Card_Disputes",
    "Digital_Banking", "Access_Issues", "Fraud_Desk", "Suspicious_Alerts",
    "General_Service", "Branch_Queries", "KYC_Team", "Compliance",
    "Account_Maintenance", "Customer_Care", "Billing_Team"
]

def random_attachment(cl):
    if random.random() < 0.35:
        return random.choice(CLUSTERS[cl]["attachments"])
    return ""

def random_sender():
    # Mix corporate, personal, spammy patterns
    domains = ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "company.com"]
    name = fake.first_name().lower() + "." + fake.last_name().lower()
    if random.random() < 0.05:
        name += str(random.randint(10,99))
    return f"{name}@{random.choice(domains)}"

def random_time():
    now = datetime.now()
    delta = timedelta(days=random.randint(0,180), hours=random.randint(0,23))
    return (now - delta).strftime("%Y-%m-%d %H:%M:%S")

def add_noise(text):
    noise_bits = [
        "", "Please revert ASAP.", "Thanks & regards.", 
        "Sent from my iPhone", "Forwarded message:", 
        "Please consider the environment before printing this email.",
        "Hi team,", "Regards,", "Pls chk.", "Attaching FYI."
    ]
    if random.random() < 0.25:
        return text + " " + random.choice(noise_bits)
    return text

def generate_email_row():
    # Pick true topic cluster
    cl = random.choice(LATENT)

    subject = random.choice(CLUSTERS[cl]["subjects"])
    body = random.choice(CLUSTERS[cl]["body_blocks"])

    # Add noise, signatures, replies
    if random.random() < 0.15:
        body = "Re: " + body
    if random.random() < 0.15:
        body = "Fwd: " + body
    body = add_noise(body)

    sender = random_sender()
    received_time = random_time()
    attachments = random_attachment(cl)

    # Map to an imperfect mailbox (noise 15%)
    if random.random() < 0.15:
        mailbox = random.choice(MAILBOXES)  # mislabel
    else:
        # biased mapping to natural groups
        if cl == "loan_queries":
            mailbox = random.choice(["Loans_Team", "Loans_Review"])
        elif cl == "credit_card_disputes":
            mailbox = random.choice(["Card_Support", "Card_Disputes", "Billing_Team"])
        elif cl == "account_access":
            mailbox = random.choice(["Digital_Banking", "Access_Issues", "Customer_Care"])
        elif cl == "fraud_alerts":
            mailbox = random.choice(["Fraud_Desk", "Suspicious_Alerts", "Compliance"])
        else:
            mailbox = random.choice(["General_Service", "Branch_Queries", "KYC_Team"])

    return [
        subject,
        body,
        sender,
        received_time,
        attachments,
        mailbox
    ]


# ----------------------------
# Generate CSV
# ----------------------------
outfile = "synthetic_emails_10k.csv"

with open(outfile, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["subject", "body", "sender", "received_time", "attachments", "mailbox"])

    for _ in range(10000):
        writer.writerow(generate_email_row())

print("Generated:", outfile)
