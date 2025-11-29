import csv
import random
from datetime import datetime, timedelta
from faker import Faker

# Initialize Faker
fake = Faker()

# --- Dataset Configuration ---
NUM_RECORDS = 50000
NUM_CATEGORIES = 150
OUTPUT_FILE = "synthetic_email_dataset.csv"

# --- Realistic Data Elements ---
DEPARTMENTS = [
    "hr", "it-support", "finance-reports", "all-staff", "legal",
    "marketing-updates", "social-committee", "project-pegasus", "engineering",
    "sales", "product-management", "customer-support", "security", "facilities"
]

SENDER_DOMAINS = ["example.com", "example.org", "example.net"]

# --- Category Definitions ---
# Define 150 categories with subject and body templates.
# Using placeholders like {product}, {date}, {ticket_id}, {name}, {department}
# which will be replaced by Faker.
CATEGORIES = {
    "Annual Performance Reviews": {
        "subjects": ["Reminder: Annual Performance Reviews Due", "Action Required: Complete Your Performance Review"],
        "bodies": ["This is a reminder that annual performance reviews are due by {date}. Please discuss with your manager.", "Your annual performance review is available in the HR portal. Please complete your self-assessment by {date}."]
    },
    "Social Media Analytics": {
        "subjects": ["Weekly Social Media Analytics", "Monthly Social Media Report"],
        "bodies": ["Here are the social media analytics for the past week. Engagement is up by {number}%.", "Attached is the monthly social media report. Our new campaign on {social_platform} is performing well."]
    },
    "IT Maintenance": {
        "subjects": ["IT ALERT: Planned System Maintenance on {day}", "Upcoming Maintenance for {system}"],
        "bodies": ["Please be advised of planned system maintenance scheduled for {day} at {time}. The {system} will be unavailable.", "We will be performing scheduled maintenance on the {system} this weekend. We apologize for any inconvenience."]
    },
    "Expense Reports": {
        "subjects": ["Action Needed: Submit Your Expense Report", "Reminder: Expense Reports Due"],
        "bodies": ["A reminder to all employees to submit their expense reports for the month of {month} by EOD Friday.", "Late expense report submissions may not be processed in time for payroll. Please submit by {date}."]
    },
}
validated_categories = {}
for i in range(NUM_CATEGORIES):
    cat_name = f"Category {i+1}"
    if i < len(list(CATEGORIES.keys())):
        cat_name = list(CATEGORIES.keys())[i]

    if cat_name in CATEGORIES:
        validated_categories[cat_name] = CATEGORIES[cat_name]
    else: # Fallback for categories not explicitly defined
        validated_categories[cat_name] = {
            "subjects": [f"Update on Project {fake.word().capitalize()}", f"Meeting Summary: {fake.bs()}"],
            "bodies": [f"This is an update regarding {fake.sentence(nb_words=5)}", f"Please review the attached document about the {fake.bs()} initiative."]
        }

# --- Data Generation ---
def generate_fake_data(writer, num_records, categories_map):
    """Generates and writes fake email data to a CSV."""
    category_names = list(categories_map.keys())
    start_date = datetime.now() - timedelta(days=365)
    end_date = datetime.now()

    for i in range(num_records):
        category = random.choice(category_names)
        templates = categories_map[category]

        # Pick a random template
        subject_template = random.choice(templates["subjects"])
        body_template = random.choice(templates["bodies"])

        # Populate template placeholders
        replacements = {
            "{date}": fake.date_this_year().strftime('%B %d, %Y'),
            "{number}": str(random.randint(5, 50)),
            "{social_platform}": random.choice(["Twitter", "Instagram", "Facebook", "LinkedIn"]),
            "{day}": fake.day_of_week(),
            "{time}": fake.time(),
            "{system}": random.choice(["JIRA", "Internal Wiki", "Email Server", "CRM"]),
            "{month}": fake.month_name(),
            "{product}": fake.word().capitalize() + "Pro",
            "{ticket_id}": str(fake.random_number(digits=4)),
            "{name}": fake.first_name(),
            "{department}": random.choice(DEPARTMENTS).replace('-', ' ').title()
        }

        subject = subject_template
        body = body_template
        # Replace placeholders individually to avoid KeyErrors for unused placeholders
        for key, value in replacements.items():
            subject = subject.replace(key, str(value))
            body = body.replace(key, str(value))

        # Create a record
        record = {
            "id": i,
            "category": category,
            "subject": subject,
            "body": body,
            "sender": f"{random.choice(DEPARTMENTS)}@{random.choice(SENDER_DOMAINS)}",
            "mailbox": f"mailbox{random.randint(1, 500)}@example.com",
            "timestamp": fake.date_time_between(start_date=start_date, end_date=end_date).isoformat()
        }
        writer.writerow(record)

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Generating {NUM_RECORDS} records across {len(validated_categories)} categories...")
    with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "category", "subject", "body", "sender", "mailbox", "timestamp"]
        csv_writer = csv.DictWriter(f, fieldnames=fieldnames)
        csv_writer.writeheader()
        generate_fake_data(csv_writer, NUM_RECORDS, validated_categories)
    print(f"Successfully created '{OUTPUT_FILE}'.")