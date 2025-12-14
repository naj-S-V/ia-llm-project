# Eco-Sorter AI 🌍♻️

**[Add a concise one-sentence description of your project here]**

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [First Run Setup](#first-run-setup)
- [Running the Application](#running-the-application)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

**[Provide a detailed description of what Eco-Sorter AI does, its goals, and the target audience here. Explain the problem it solves and its key features.]**

### Key Features

- **[Feature 1]**: [Description]
- **[Feature 2]**: [Description]
- **[Feature 3]**: [Description]
- **[Add more features as needed]**

## Architecture

**[Describe the system architecture here. Include diagrams if applicable, explain the main components, and how they interact.]**

### Main Components

- **Vision Model (YOLO)**: Detects and classifies waste objects from images
- **RAG Engine**: Retrieves region-specific waste sorting guidelines
- **LLM Agent**: Provides intelligent recommendations using Mistral AI
- **Streamlit Frontend**: User-friendly interface for interaction

## Prerequisites

Before running this project, ensure you have the following installed:

- **Python 3.11+**
- **pip** (Python package manager)
- **Git** (for version control)

### Required API Keys

You will need:
- **Mistral API Key**: [Get it from Mistral AI console](https://console.mistral.ai/)

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/naj-S-V/ia-llm-project.git
cd ia-llm-project
```

### Step 2: Create a Virtual Environment

```bash
# On Windows
python -m venv .venv
.venv\Scripts\activate

# On macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file in the project root and add your Mistral API key:

```bash
# .env file
MISTRAL_API_KEY=your_mistral_api_key_here
```

**⚠️ Never commit the `.env` file to version control!**

## First Run Setup

### Build the Vector Database

The application uses a vector database (Chroma) to store region-specific waste sorting guidelines. You need to generate this database **once** before running the application:

```bash
# Activate your virtual environment first (if not already activated)
.venv\Scripts\activate

# Run the setup script to create the vector database
python utils/setup_vectordb.py
```

**What this script does:**
- Loads all regional guides (Antwerp, Brussels, Liège, etc.)
- Splits documents into semantic chunks
- Generates embeddings using HuggingFace models
- Stores everything in `data/vectorstore/` for fast retrieval

**⏱️ This may take a few minutes on the first run.** The embeddings are cached locally, so subsequent runs will be faster.


## Running the Application

Once setup is complete, launch the Streamlit app:

```bash
# Make sure your virtual environment is activated
.venv\Scripts\activate

# Run the Streamlit application
streamlit run src/app.py
```

The application will open in your default browser at `http://localhost:8501`

### Important Notes

- **Do NOT re-run `setup_vectordb.py`** every time you start the app (this would recreate the database unnecessarily)
- The vector database is persistent and stored in `data/vectorstore/`
- To update guidelines, simply re-run the setup script and restart the app

## Project Structure

```
ia-llm-project/
├── data/
│   ├── dataset/                    # Waste image datasets for training
│   ├── documents/                  # Regional waste sorting guides
│   ├── vectorstore/                # Chroma vector database (generated)
│   └── outputs/                    # Inference results
├── models_training_runs/           # Trained YOLO model checkpoints
├── notebooks/                      # Jupyter notebooks for exploration
├── scripts/
│   └── setup_vectordb.py          # Vector database initialization script
├── src/
│   ├── app.py                     # Streamlit application entry point
│   ├── agent_logic.py             # RAG chain and LLM integration
│   ├── vision_model.py            # YOLO object detection model
│   ├── tools.py                   # Utility functions
│   └── __init__.py
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables (not in repo)
├── .gitignore                     # Git ignore rules
└── README.md                      # This file
```

## Usage

**[Provide detailed usage instructions and examples here. Include screenshots or examples of how users interact with the application.]**

### Example Interaction

```
User: "I have a plastic bag. Where should I throw it in Brussels?"
Eco-Sorter AI: "In Brussels, plastic bags should go in the [Yellow Bag]. 
This includes..."
```

### Supported Regions

- Antwerp (Antwerpen)
- Brussels (Bruxelles)
- Liège
- Namur
- Hainaut
- Brabant Wallon
- Charleroi
- Luxembourg

## Contributing

**[Describe how others can contribute to this project. Include guidelines for pull requests, coding standards, etc.]**

## License

**[Specify the license under which this project is distributed, e.g., MIT, Apache 2.0, etc.]**

---

## Troubleshooting

**[Add common issues and their solutions here as the project develops]**

### Issue: "MISTRAL_API_KEY not found"
- Ensure you created a `.env` file in the project root
- Verify the API key is correct
- Restart the Streamlit app after updating `.env`

### Issue: "Vector database not found"
- Run `python scripts/setup_vectordb.py` to initialize the database
- Ensure `data/vectorstore/` directory exists after running

---

**Last Updated**: December 14, 2025  
**Maintainers**: [Your Name/Team]
