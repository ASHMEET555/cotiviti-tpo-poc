# Submission Checklist & Instructions

## Before You Submit

- [ ] Run `python scripts/train_all.py` – all models trained successfully
- [ ] Run `streamlit run src/app/streamlit_app.py` – dashboard launches, all tabs work
- [ ] Run `python scripts/make_assets.py` – charts exported to `assets/charts/`
- [ ] Run `python deliverables/report/generate_report.py` – `Report.docx` created
- [ ] Run `python deliverables/slides/generate_slides.py` – `Presentation.pptx` created
- [ ] Open `Presentation.pptx` – replace `<Your Full Name>` and `<Your University>` placeholders
- [ ] Open `Report.docx` – replace `<Your Full Name>` and `<Your University>` placeholders
- [ ] Record video per `deliverables/video/recording_guide.md`
- [ ] Video saved as `deliverables/video/<YourName>_TPO_Sentinel.mp4`

---

## GitHub Setup Commands

Run in your project root:

```bash
# 1. Initialize git repository
git init
git add .
git commit -m "Initial submission – Cotiviti TPO Sentinel POC"

# 2. Create public GitHub repo (requires GitHub CLI: https://cli.github.com/)
gh repo create cotiviti-tpo-poc --public --source=. --remote=origin --push

# 3. Add the Cotiviti evaluator as a collaborator
#    (replace YOUR_GITHUB_USERNAME with your actual username)
gh api repos/YOUR_GITHUB_USERNAME/cotiviti-tpo-poc/collaborators/jesus.hurtado \
  --method PUT -f permission=read
```

**Without GitHub CLI:**
1. Go to https://github.com/new
2. Create a new repository named `cotiviti-tpo-poc` (Public)
3. Follow the "push an existing repository" instructions
4. Go to Settings → Collaborators → Add `jesus.hurtado` (read access)

---

## Email Template

**To:** jesus.hurtado@cotiviti.com  
**Subject:** `INTERN - <Your Full Name> - <Your University>`

```
Dear Jesus,

Please find my Cotiviti Intern Assessment submission below.

Topic Selected: #2 – Clinical Decision Making and Pattern Recognition
in Health Care (TPO)

GitHub Repository: https://github.com/<YOUR_GITHUB_USERNAME>/cotiviti-tpo-poc

The repository contains:
  • POC demo code (Python/Streamlit) – run: python scripts/train_all.py && streamlit run src/app/streamlit_app.py
  • Written report (deliverables/report/Report.docx)
  • PowerPoint presentation (deliverables/slides/Presentation.pptx)
  • Video recording (deliverables/video/)

I look forward to hearing from you.

Best regards,
<Your Full Name>
<Your University>
<Your Email>
<Your Phone>
```
