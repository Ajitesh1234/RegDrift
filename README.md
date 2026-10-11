# RegDrift
Self-healing compliance layer: reads a regulator circular (txt/PDF/URL), finds which RPA bots (UiPath / Blue Prism style) it affects, and files Jira tickets. Reviewer decisions become LoRA training data.

Pipeline (LangGraph): ingest -> obligation extraction -> RAG impact analysis (ChromaDB) -> gap reasoning -> Jira -> human review -> LoRA retrain.

## Run (no local install)
- Colab: `RegDrift_Colab.ipynb` (pipeline), `RegDrift_LoRA_Train.ipynb` (fine-tune, T4 GPU).
- GitHub Actions: Actions > "RegDrift run" > Run workflow (runs tests + pipeline, uploads tickets.json).
- Optional secrets: GROQ_API_KEY, JIRA_URL, JIRA_EMAIL, JIRA_TOKEN, JIRA_PROJECT, LORA_ADAPTER.
  Without them it runs offline in dry-run mode.

## Commands
    python regdrift.py circular.pdf          # or a .txt or an https:// link to a circular
    python review.py                         # approve/reject tickets -> data/feedback.jsonl
    python build_lora_data.py                # seed + feedback -> data/train.jsonl
    python -m unittest -v tests/test_pipeline.py

## Honest limits
- Sample circular and bots are synthetic demo data. Seed LoRA examples (13) are hand-written; add real ones.
- Deterministic checks only work where a bot exposes the rule's parameter; otherwise the LLM or a manual-review flag decides.
- Blue Prism/UiPath are represented by bot metadata, not live bots.

RegDrift creates Jira tickets automatically for each non-compliant bot:
<img width="1447" height="198" alt="image" src="https://github.com/user-attachments/assets/5cf40fc9-61f6-4c3c-b86b-ba95f8328ff4" />
RegDrift creates Jira tickets automatically for each non-compliant bot:
<img width="1376" height="148" alt="image" src="https://github.com/user-attachments/assets/87dc4940-97f1-4092-ab6a-cef2301ba317" />

