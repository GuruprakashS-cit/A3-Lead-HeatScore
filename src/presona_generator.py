# src/generate_persona_snippets_realistic.py
import pandas as pd
import random

# Roles and corresponding interest templates
role_snippet_map = {
    "Student": [
        "Interested in online courses and hands-on projects.",
        "Looks for tutorials and study materials.",
        "Enjoys interactive learning and webinars."
    ],
    "Manager": [
        "Seeks team training and leadership workshops.",
        "Interested in strategy and productivity guides.",
        "Follows industry trends for decision making."
    ],
    "Developer": [
        "Likes coding tutorials and practical projects.",
        "Follows tech blogs and open-source contributions.",
        "Engages with AI/ML development guides."
    ],
    "Analyst": [
        "Enjoys data analysis, dashboards, and reports.",
        "Interested in insights and visualization tools.",
        "Follows market and business analytics trends."
    ],
    "Designer": [
        "Seeks creative workshops and design tutorials.",
        "Likes UI/UX case studies and inspirations.",
        "Follows design trends and innovative ideas."
    ],
    "Engineer": [
        "Interested in practical engineering projects.",
        "Follows tech research and innovation news.",
        "Likes building prototypes and experiments."
    ],
    "Consultant": [
        "Seeks actionable insights for clients.",
        "Interested in industry best practices.",
        "Follows market trends and solution strategies."
    ],
    "Entrepreneur": [
        "Looks for startup guidance and mentorship.",
        "Interested in growth hacks and productivity tips.",
        "Follows funding trends and business ideas."
    ],
    "Researcher": [
        "Interested in academic papers and experiments.",
        "Follows latest research in AI/ML and data science.",
        "Likes technical documentation and case studies."
    ],
    "Intern": [
        "Seeks learning opportunities and mentorship.",
        "Interested in practical experience and tasks.",
        "Looks for guidance on skill-building projects."
    ]
}

# Generate 20 persona snippets
data = []
for i in range(20):
    role = random.choice(list(role_snippet_map.keys()))
    snippet = random.choice(role_snippet_map[role])
    data.append({"role": role, "snippet": snippet})

# Save to CSV
persona_file = "../data/persona_snippets.csv"
df = pd.DataFrame(data)
df.to_csv(persona_file, index=False)
print(f"Realistic persona snippets CSV created at {persona_file}")
