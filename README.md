# Member Question-Answering Service

Lightweight FastAPI service that answers natural-language questions about member data provided by a public `/messages` API.

## Quick start (local)

```bash
# clone repo
git clone git@github.com:<your-username>/aiml_engineer_assignment.git
cd aiml_engineer_assignment

# create virtualenv (optional)
python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
