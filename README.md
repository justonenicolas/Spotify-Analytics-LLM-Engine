# Spotify Analytics Assistant 🎵📊
An interactive natural language analytics assistant built with **Streamlit**, **Plotly**, **Pandas**, and the **OpenAI API**. This application translates natural language queries about streaming habits into dynamically executed Python/Plotly code, visualizing trends and insights directly from a local Spotify streaming dataset.

---

## 🌟 Key Features
- **Natural Language Data Exploration**: Ask complex questions about your listening habits (e.g., top artists, seasonal listening trends, skip counts, hourly usage) in natural prose.
- **Privacy-Preserving & Cost-Optimized Architecture**: Raw streaming data remains local. The LLM only receives dataset metadata, valid boundaries, and column schemas to generate code execution strings.
- **Dynamic Code Execution**: Uses `exec()` in a controlled scope to execute Plotly code generated on-the-fly by the LLM.
- **Out-of-Scope & Constraint Guardrails**: Employs strict system prompts to prevent hallucinatory queries, unhandled columns, out-of-range date requests, or cross-conversational context leaks.
- **Password Protection**: Includes simple session-based authentication to secure access to the interface.

---

## 🏗️ Architecture Overview
1. **Prompt Injection & Context Validation**: Dynamic metadata (min/max timestamps, valid platform options, column names, start/end reason domains) is injected into the system prompt upon loading.
2. **LLM Query Processing**: The user's query is dispatched alongside the prompt to `gpt-4.1-mini` with low temperature (`0.2`).
3. **Structured Response**: The model responds with standard JSON specifying either an out-of-scope reason or a self-contained snippet generating a Plotly `fig` object.
4. **Local Execution**: The application parses the response, executes the chart-generation code using local execution variables (`df`, `pd`, `px`, `go`), and renders both the interactive chart and interpretation.

---

## Installation
### Clone the repository:
git clone [https://github.com/](https://github.com/)<your-username>/<your-repo-name>.git
cd <your-repo-name>

### Install required packages
pip install -r requirements.txt

### Configure Secrets:
- Create a .streamlit/secrets.toml file in the project root folder with the following contents:
OPENAI_API_KEY = "your-openai-api-key"
PASSWORD = "your-app-password"

---

## 🛠️ Project Structure

```text
.
├── app.py                   # Main Streamlit application and execution logic
├── streaming_history.json   # Spotify streaming history dataset (local)
├── requirements.txt         # Dependencies list
└── README.md                # Project documentation
