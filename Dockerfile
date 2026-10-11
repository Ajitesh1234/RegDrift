FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -m regdrift && chown -R regdrift /app
USER regdrift
# Secrets (GROQ_API_KEY, JIRA_*) come from the environment, never from the image.
ENTRYPOINT ["python", "regdrift.py"]
CMD ["--dry-run"]
