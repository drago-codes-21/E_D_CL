import pandas as pd
import random
import datetime
import os

# --- Configuration ---
NUM_RECORDS = 10000
NUM_MAILBOXES = 50
OUTPUT_PATH = "data/raw/synthetic_emails.csv"
LOOKBACK_DAYS = 180

# --- Content Generation ---

# 50 mailboxes
MAILBOXES = [f"mailbox{i}@example.com" for i in range(NUM_MAILBOXES)]

# Senders
SENDERS = [
    "hr@example.com", "it-support@example.com", "finance-reports@example.com",
    "marketing-updates@example.com", "project-pegasus@example.com", "legal@example.com",
    "all-staff@example.com", "social-committee@example.com"
]

# Topics and associated keywords/phrases
TOPICS = {
    "HR": {
        "subjects": ["Important: New Company Policy on {topic}", "Reminder: Annual Performance Reviews", "Welcome to the team, {name}!", "Action Required: Complete Your Benefits Enrollment"],
        "bodies": [
            "Please review the updated company policy on {topic}, attached for your convenience. All employees are required to acknowledge receipt by EOD Friday.",
            "This is a reminder that annual performance reviews are due by {date}. Please schedule a meeting with your manager to discuss your self-assessment.",
            "We are thrilled to welcome {name} to the team as our new {role}! Please take a moment to introduce yourself.",
            "Open enrollment for benefits is now active. Please log in to the HR portal to make your selections before the deadline on {date}."
        ],
        "keywords": {
            "topic": ["Remote Work", "Expense Reporting", "Paid Time Off"],
            "name": ["Alice", "Bob", "Charlie", "Diana"],
            "role": ["Software Engineer", "Marketing Analyst", "Data Scientist"],
            "date": ["October 31st", "November 15th", "December 1st"]
        }
    },
    "IT": {
        "subjects": ["IT ALERT: Planned System Maintenance on {date}", "URGENT: Phishing Attempt Detected", "Your Support Ticket #{ticket_id} has been updated", "Scheduled Downtime for {service}"],
        "bodies": [
            "Please be advised of planned system maintenance scheduled for {date} at {time}. The {service} will be unavailable during this time. We apologize for any inconvenience.",
            "Our security systems have detected a phishing attempt targeting our employees. DO NOT click on any links from suspicious emails. If you receive a suspicious email, please forward it to it-support@example.com.",
            "Your support ticket #{ticket_id} regarding '{issue}' has been updated. A technician has been assigned and will contact you shortly.",
            "The {service} will be down for a scheduled update this weekend. We expect services to be restored by Monday morning."
        ],
        "keywords": {
            "date": ["Saturday", "Sunday", "Friday night"],
            "time": ["10:00 PM UTC", "2:00 AM UTC"],
            "service": ["JIRA", "Confluence", "Internal Wiki", "VPN"],
            "ticket_id": [random.randint(1000, 9999) for _ in range(5)],
            "issue": ["Printer not working", "Cannot access shared drive", "Password reset request"]
        }
    },
    "FINANCE": {
        "subjects": ["Q{q} Financial Report Attached", "Action Needed: Submit Your Expense Report by {date}", "Budget Planning for Q{next_q} {year}"],
        "bodies": [
            "Attached is the financial summary for Q{q} of {year}. Please review and provide any feedback by the end of the week.",
            "A reminder to all employees to submit their expense reports for the month of {month} by {date}. Late submissions may not be processed in time for payroll.",
            "We are beginning the budget planning process for Q{next_q} {year}. Please submit your department's budget proposal to the finance team."
        ],
        "keywords": {
            "q": [1, 2, 3, 4],
            "next_q": [2, 3, 4, 1],
            "year": [2025, 2024],
            "month": ["October", "November", "September"],
            "date": ["the 5th of next month", "EOD Friday"]
        }
    },
    "MARKETING": {
        "subjects": ["New Marketing Campaign Launch: {campaign}", "Weekly Social Media Analytics", "Creative Brief for the {product} Launch"],
        "bodies": [
            "We are excited to announce the launch of our new marketing campaign, '{campaign}'! See the attached brief for more details on the target audience and messaging.",
            "Here are the social media analytics for the past week. Engagement is up by {percent}% on Twitter, and our new Instagram posts are performing well.",
            "The creative brief for the upcoming {product} launch is ready for review. We need all creative assets finalized by the end of the month."
        ],
        "keywords": {
            "campaign": ["Project Aurora", "Operation Sunshine", "Summer Sale"],
            "percent": [random.randint(5, 25) for _ in range(3)],
            "product": ["WidgetX", "GizmoPro", "ServicePlus"]
        }
    }
}

# --- Data Generation Function ---
def generate_record():
    topic_name = random.choice(list(TOPICS.keys()))
    topic_data = TOPICS[topic_name]

    subject_template = random.choice(topic_data["subjects"])
    body_template = random.choice(topic_data["bodies"])

    # Fill templates with random keywords
    context = {}
    for key, values in topic_data["keywords"].items():
        context[key] = random.choice(values)

    subject = subject_template.format(**context)
    body = body_template.format(**context)

    # Other fields
    sender = random.choice(SENDERS)
    mailbox = random.choice(MAILBOXES)
    now = datetime.datetime.now(datetime.timezone.utc)
    received_time = now - datetime.timedelta(
        days=random.randint(0, LOOKBACK_DAYS),
        hours=random.randint(0, 23),
        minutes=random.randint(0, 59)
    )
    attachments = f"attachment_{random.randint(1,100)}.pdf" if random.random() > 0.8 else ""

    return {
        "subject": subject,
        "body": body,
        "sender": sender,
        "received_time": received_time.isoformat(),
        "attachments": attachments,
        "mailbox": mailbox
    }

# --- Main Script ---
if __name__ == "__main__":
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    records = [generate_record() for _ in range(NUM_RECORDS)]
    df = pd.DataFrame(records)

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Successfully generated {NUM_RECORDS} records to {OUTPUT_PATH}")
