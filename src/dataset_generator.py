# src/dataset_generator.py

import pandas as pd
import numpy as np
from faker import Faker
import random
import os

# -----------------------
# Ensure data directory exists
# -----------------------
os.makedirs("../data", exist_ok=True)

# -----------------------
# Initialize Faker
# -----------------------
fake = Faker()

# -----------------------
# Parameters
# -----------------------
NUM_ROWS = 2000
OUTPUT_FILE = "../data/leads.csv"
DICT_FILE = "../data/dictionary.md"

# -----------------------
# Feature options
# -----------------------
sources = ["Webinar", "LinkedIn Ad", "Referral", "Organic", "Cold Email"]
regions = ["India", "US", "Europe", "SEA"]
roles = ["Student", "Working Professional", "Manager", "Executive"]
campaigns = ["AI_Course_2025", "DataScience_Offer", "Summer_Internship"]
last_touches = ["Website Visit", "Downloaded Brochure", "Attended Webinar", "Emailed Support"]

# -----------------------
# Generate data
# -----------------------
data = []

for i in range(1, NUM_ROWS + 1):
    lead_id = i
    source = random.choice(sources)
    recency_days = np.random.randint(0, 91)  # 0–90 days
    region = random.choice(regions)
    role = random.choice(roles)
    campaign = random.choice(campaigns)
    page_views = np.random.randint(0, 101)  # 0–100
    last_touch = random.choice(last_touches)
    prior_course_interest = np.random.choice([0, 1])
    
    # -----------------------
    # Assign heat_label based on rules
    # -----------------------
    score = 0
    if recency_days <= 7:
        score += 2
    elif recency_days <= 30:
        score += 1

    if page_views >= 20:
        score += 2
    elif page_views >= 10:
        score += 1

    if source in ["Webinar", "Referral"]:
        score += 1

    if prior_course_interest == 1:
        score += 1

    # Map score to HeatLabel
    if score >= 5:
        heat_label = "Hot"
    elif score >= 3:
        heat_label = "Warm"
    else:
        heat_label = "Cold"

    data.append([lead_id, source, recency_days, region, role, campaign, page_views, last_touch, prior_course_interest, heat_label])

# -----------------------
# Create DataFrame
# -----------------------
columns = ["lead_id", "source", "recency_days", "region", "role", "campaign", "page_views", "last_touch", "prior_course_interest", "heat_label"]
df = pd.DataFrame(data, columns=columns)

# Save CSV
df.to_csv(OUTPUT_FILE, index=False)
print(f"Synthetic dataset saved to {OUTPUT_FILE}")

# -----------------------
# Create dictionary.md
# -----------------------
with open(DICT_FILE, "w") as f:
    f.write("# Feature Dictionary\n\n")
    f.write("**lead_id**: Unique identifier for each lead\n")
    f.write("**source**: How the lead came to the platform (Webinar, LinkedIn Ad, etc.)\n")
    f.write("**recency_days**: Days since last interaction\n")
    f.write("**region**: Geographic region of the lead\n")
    f.write("**role**: Professional role of the lead\n")
    f.write("**campaign**: Marketing campaign that captured the lead\n")
    f.write("**page_views**: Number of pages visited on the site\n")
    f.write("**last_touch**: Last action by the lead (Webinar, Email, etc.)\n")
    f.write("**prior_course_interest**: Boolean if lead showed prior interest in courses\n")
    f.write("**heat_label**: Target label (Hot/Warm/Cold) assigned using simple scoring rules\n")
print(f"Dictionary file saved to {DICT_FILE}")
