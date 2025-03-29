import json
import re
import fitz  # PyMuPDF
from collections import Counter

# Predefined mapping for keyword normalization
KEYWORDS_MAP = {
    # Programming Languages
    "python": "python", "java": "java", "c": "c", "c++": "c++", "cpp": "c++",
    "c#": "csharp", "csharp": "csharp", "javascript": "javascript", "js": "javascript",
    "typescript": "typescript", "ts": "typescript", "html": "html", "css": "css",
    "scss": "scss", "sass": "sass", "less": "less", "sql": "sql", "r": "r",
    "go": "golang", "golang": "golang", "rust": "rust", "swift": "swift",
    "kotlin": "kotlin", "php": "php", "ruby": "ruby", "perl": "perl",
    "scala": "scala", "dart": "dart", "objective-c": "objective-c",
    "objective c": "objective-c", "objc": "objective-c", "bash": "bash",
    "shell scripting": "shell-scripting", "shellscripting": "shell-scripting",
    "powershell": "powershell", "lua": "lua", "groovy": "groovy", "elixir": "elixir",
    "haskell": "haskell", "clojure": "clojure", "f#": "fsharp", "erlang": "erlang",
    "matlab": "matlab", "vba": "vba", "cobol": "cobol", "fortran": "fortran",
    "assembly": "assembly", "racket": "racket", "scheme": "scheme",
    "lisp": "lisp", "pl/sql": "plsql", "plsql": "plsql",
    "t-sql": "tsql", "tsql": "tsql",

    # AI / ML / Data Science
    "machine learning": "ml", "machinelearning": "ml", "ml": "ml",
    "deep learning": "dl", "deeplearning": "dl", "dl": "dl",
    "artificial intelligence": "ai", "artificialintelligence": "ai", "ai": "ai",
    "reinforcement learning": "rl", "reinforcementlearning": "rl", "rl": "rl",
    "pandas": "pandas", "numpy": "numpy", "tensorflow": "tensorflow",
    "tf": "tensorflow", "pytorch": "pytorch", "torch": "pytorch",
    "scikit-learn": "sklearn", "scikitlearn": "sklearn", "sklearn": "sklearn",
    "xgboost": "xgboost", "lightgbm": "lightgbm", "catboost": "catboost",
    "opencv": "opencv", "computer vision": "cv", "computervision": "cv", "cv": "cv",
    "nltk": "nltk", "spacy": "spacy", "huggingface": "huggingface",
    "fine-tune": "fine-tuning", "finetune": "fine-tuning", 
    "fine tuning": "fine-tuning", "finetuning": "fine-tuning",
    "transfer learning": "transfer-learning", "transferlearning": "transfer-learning",
    "object detection": "object-detection", "objectdetection": "object-detection",
    "image classification": "image-classification", "imageclassification": "image-classification",
    "yolo": "yolo", "transformers": "transformers", "bert": "bert",
    "gpt": "gpt", "lstm": "lstm", "gan": "gan", "autoencoders": "autoencoders",
    "speech recognition": "speech-recognition", "speechrecognition": "speech-recognition",
    "ocr": "ocr", "text-to-speech": "tts", "text to speech": "tts", "tts": "tts",
    "speech-to-text": "stt", "speech to text": "stt", "stt": "stt",
    "time series analysis": "time-series-analysis", "timeseriesanalysis": "time-series-analysis",
    "anomaly detection": "anomaly-detection", "anomalydetection": "anomaly-detection",
    "recommendation systems": "recommendation-systems", "recommendationsystems": "recommendation-systems",
    "generative ai": "gen-ai", "generativeai": "gen-ai",
    "self-supervised learning": "self-supervised-learning", "selfsupervisedlearning": "self-supervised-learning",
    "semi-supervised learning": "semi-supervised-learning", "semisupervisedlearning": "semi-supervised-learning",
    "unsupervised learning": "unsupervised-learning", "unsupervisedlearning": "unsupervised-learning",
    "feature engineering": "feature-engineering", "featureengineering": "feature-engineering",
    "hyperparameter tuning": "hyperparameter-tuning", "hyperparametertuning": "hyperparameter-tuning",
    "bayesian optimization": "bayesian-optimization", "bayesianoptimization": "bayesian-optimization",

    # DevOps / Cloud
    "git": "git", "github": "github", "git hub": "github", "gitlab": "gitlab",
    "git lab": "gitlab", "bitbucket": "bitbucket", "bit bucket": "bitbucket",
    "docker": "docker", "kubernetes": "kubernetes", "k8s": "kubernetes",
    "jenkins": "jenkins", "ansible": "ansible", "terraform": "terraform",
    "ci/cd": "cicd", "cicd": "cicd", "aws": "aws",
    "amazon web services": "aws", "amazonwebservices": "aws",
    "azure": "azure", "microsoft azure": "azure", "microsoftazure": "azure",
    "gcp": "gcp", "google cloud": "gcp", "googlecloud": "gcp",
    
    # Blockchain / Web3
    "blockchain": "blockchain", "smart contracts": "smart-contracts",
    "smartcontracts": "smart-contracts", "ethereum": "ethereum",
    "solidity": "solidity", "web3.js": "web3", "web3js": "web3", "web3": "web3",
    
    # Soft Skills & Management
    "leadership": "leadership", "mentoring": "mentoring", "coaching": "coaching",
    "public speaking": "public-speaking", "publicspeaking": "public-speaking",
    "problem solving": "problem-solving", "problemsolving": "problem-solving",
    "critical thinking": "critical-thinking", "criticalthinking": "critical-thinking",
    "time management": "time-management", "timemanagement": "time-management",
    
    # Miscellaneous
    "excel": "excel", "spreadsheet": "spreadsheet", "vba": "vba",
    "google sheets": "googlesheets", "googlesheets": "googlesheets",
    "regex": "regex", "shell scripting": "shell-scripting",
    "json": "json", "xml": "xml", "csv": "csv"
}

def normalize_keyword(keyword):
    """Convert variations of a keyword to its canonical form."""
    return KEYWORDS_MAP.get(keyword.lower(), keyword.lower())

def extract_keywords(text):
    """Extract and normalize keywords from text."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s\-\+]', ' ', text)  # Remove special characters
    keyword_counter = Counter()

    # Check for multi-word keywords first
    for keyword_variant in sorted(KEYWORDS_MAP.keys(), key=lambda x: -len(x.split())):
        occurrences = len(re.findall(r'\b' + re.escape(keyword_variant) + r'\b', text))
        if occurrences:
            normalized = normalize_keyword(keyword_variant)
            keyword_counter[normalized] += occurrences

    return keyword_counter

def extract_resume_text(resume):
    """Convert resume JSON into a single text block."""
    return json.dumps(resume, ensure_ascii=False).lower()  # Convert entire JSON to lowercase text

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    return text

def calculate_match_percentage(resume_keywords, jd_keywords):
    """Calculate percentage match between resume and job description."""
    common_keywords = set(resume_keywords.keys()).intersection(jd_keywords.keys())
    if not jd_keywords:
        return 0
    return round((len(common_keywords) / len(jd_keywords)) * 100, 2)

def main():
    resume_json = "resume_data.json"  # Path to resume JSON file
    jd_pdf_file = "jd.pdf"  # Path to job description PDF file

    # Load resume JSON
    with open(resume_json, "r", encoding="utf-8") as file:
        resume = json.load(file)

    # Extract text from resume and JD
    resume_text = extract_resume_text(resume)
    jd_text = extract_text_from_pdf(jd_pdf_file)

    # Extract and normalize keywords
    resume_keywords = extract_keywords(resume_text)
    jd_keywords = extract_keywords(jd_text)

    # Calculate match percentage
    match_percentage = calculate_match_percentage(resume_keywords, jd_keywords)

    # Identify missing keywords
    missing_keywords = set(jd_keywords.keys()) - set(resume_keywords.keys())

    # Display results
    print(f"Resume match percentage with job description: {match_percentage}%")
    print("\nCommon Skills/Technologies:")
    for keyword, count in resume_keywords.items():
        if keyword in jd_keywords:
            print(f"{keyword}: {count} times")

    print("\nSkills in JD but NOT in Resume:")
    for keyword in missing_keywords:
        print(f"{keyword}: {jd_keywords[keyword]} times")

if __name__ == "__main__":
    main()
