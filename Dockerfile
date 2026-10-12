FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt requirements-web.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-web.txt
COPY . .
RUN useradd -m regdrift && chown -R regdrift /app
USER regdrift
# Secrets (GROQ_API_KEY, JIRA_*, APP_ACCESS_CODE) come from the environment, never from the image.
# Default: the web app. For the command line, override the command:
#   docker run --rm -e GROQ_API_KEY regdrift python regdrift.py circulars --dry-run
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uvicorn webapp.app:app --host 0.0.0.0 --port ${PORT}"]
