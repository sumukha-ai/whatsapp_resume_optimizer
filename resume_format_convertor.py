import json
import re

def clean_url(url):
    if not url:
        return ""
    url = url.replace("\\/", "/").strip()
    if not url.startswith("http"):
        url = "https://" + url
    return url

def parse_resume_data(personal_edu_about_json, projects_json, work_exp_json, output_path="resume.json"):
    # Load JSON strings
    personal_data = json.loads(personal_edu_about_json)
    projects_data = json.loads(projects_json)
    work_data = json.loads(work_exp_json)

    # -------- Contact Info --------
    full_name = personal_data.get("screen_0_Full_Name_0", "")
    email = personal_data.get("screen_0_Personal_Email_1", "")
    phone = personal_data.get("screen_0_Phone_Number_2", "")
    github = clean_url(personal_data.get("screen_0_Github_Link_3", ""))
    linkedin = clean_url(personal_data.get("screen_0_LinkedIn_Link_4", ""))
    address = personal_data.get("screen_0_Address_5", "")

    # Basic address breakdown
    address_parts = re.split(r'[, ]+', address)
    location = {
        "City": address_parts[1] if len(address_parts) > 1 else "",
        "State": address_parts[2] if len(address_parts) > 2 else "",
        "Country": address_parts[3] if len(address_parts) > 3 else "India"
    }

    contact_info = {
        "Full Name": full_name,
        "Phone Numbers": {"Mobile": phone},
        "Email": email,
        "Location": location,
        "LinkedIn": linkedin,
        "GitHub": github
    }

    # -------- Education --------
    education = []

    # Postgrad (if exists)
    if "screen_3_Name_of_Institute_0" in personal_data:
        education.append({
            "school": personal_data["screen_3_Name_of_Institute_0"],
            "degree": f"{personal_data['screen_3_Name_of_Course_1']} in {personal_data['screen_3_Name_of_Branch_2']}",
            "date": f"Expected: {personal_data['screen_3_Pass_Out_Year_4']}",
            "details": f"Current CGPA: {personal_data['screen_3_Overall_GPA_CGPA_3']}"
        })

    # Undergrad
    education.append({
        "school": personal_data["screen_2_Name_of_Institute_0"],
        "degree": f"{personal_data['screen_2_Name_of_Course_1']} in {personal_data['screen_2_Name_of_Branch_2']}",
        "date": f"Expected: {personal_data['screen_2_Pass_Out_Year_4']}",
        "details": f"Current CGPA: {personal_data['screen_2_Overall_GPA_CGPA_3']}"
    })

    # 12th / High School
    education.append({
        "school": personal_data["screen_1_Name_of_Institute_0"],
        "degree": "Intermediate",
        "date": f"Passed in {personal_data['screen_1_Pass_Out_Year_2']}",
        "details": f"Overall Percentage: {personal_data['screen_1_Percentage_GPA_1']}"
    })

    # -------- Career Objective --------
    career_objective = personal_data.get("screen_4_About_Yourself_0", "")

    # -------- Interests --------
    interests_raw = personal_data.get("screen_4_Hobbies__Interests_1", "")
    interests = [i.strip() for i in interests_raw.split(",") if i.strip()]

    # -------- Projects --------
    projects = []
    for i in range(5):
        prefix = f"screen_{i}_"
        if f"{prefix}Name_0" in projects_data:
            projects.append({
                "title": projects_data.get(f"{prefix}Name_0", ""),
                "description": projects_data.get(f"{prefix}Brief__2", ""),
                "technologies": [tech.strip() for tech in projects_data.get(f"{prefix}Technologies_Used_3", "").split(",")],
                "github": clean_url(projects_data.get(f"{prefix}Github_Link_1", ""))
            })

    # -------- Work Experience --------
    work_experience = []
    for i in range(5):
        prefix = f"screen_{i}_"
        if f"{prefix}Name_0" in work_data:
            work_experience.append({
                "company": work_data.get(f"{prefix}Name_0", ""),
                "position": work_data.get(f"{prefix}Job_Title_1", ""),
                "date": work_data.get(f"{prefix}Time_Period_2", ""),
                "description": work_data.get(f"{prefix}About_Your_Role_4", ""),
                "technologies": [tech.strip() for tech in work_data.get(f"{prefix}Technologies_Used_3", "").split(",")]
            })

    # -------- Final Resume JSON --------
    resume_json = {
        "Contact Information": contact_info,
        "Education": education,
        "Career Objective": career_objective,
        "Projects": projects,
        "Work Experience": work_experience,
        "Interests": interests
    }

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(resume_json, f, indent=2)
    print(f"✅ Resume saved to {output_path}")
