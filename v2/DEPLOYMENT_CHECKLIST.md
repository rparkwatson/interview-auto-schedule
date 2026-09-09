# Interview Scheduler v2 deployment checklist

This checklist keeps the current scheduler available while v2 is reviewed and
approved. Do not delete or reconfigure the production Streamlit app during the
review release.

## Recorded production baseline

Recorded on September 9, 2026:

- Production URL: <https://xpz4vfteutmck4ysqthrab.streamlit.app/>
- Python version: 3.11
- Access: reachable without a viewer sign-in
- Current experience: legacy paired-interviewer scheduler
- Expected GitHub source: `main` and root `app.py`; confirm these coordinates in
  Streamlit before the production cutover

Never copy Streamlit secrets into a pull request, issue, screenshot, or this
document.

## Review release

- [ ] Confirm `.gitignore` excludes local environments, secrets, uploaded source
      workbooks, and generated schedule workbooks.
- [ ] Review every file included in the v2 pull request.
- [ ] Confirm root `app.py` is unchanged by the review-release pull request.
- [ ] Confirm the `Interview Scheduler v2 checks` workflow passes on Python 3.11.
- [ ] Deploy a separate Streamlit review app using:
  - Repository: `rparkwatson/interview-auto-schedule`
  - Branch: `codex/scheduler_v2_migration`
  - Main file: `v2/app.py`
  - Python: 3.11
- [ ] Keep the existing production URL and app settings unchanged.

## Hosted acceptance

- [ ] The review app opens without an error and its cloud log is clean.
- [ ] Student and Adcom example workbooks import successfully.
- [ ] Interview periods and availability totals are correct.
- [ ] Direct interview-count entry works.
- [ ] The interview-period worksheet downloads, accepts completed counts, and
      uploads successfully.
- [ ] A standard schedule is created successfully.
- [ ] A failed standard attempt presents the intended exception choices.
- [ ] An approved exception schedule identifies affected interviewers by name.
- [ ] Stale results clear after scheduling information changes.
- [ ] All five progress-guide links reach their available sections.
- [ ] The full and simplified schedule workbooks download and agree.
- [ ] A representative 65-person, 150-period case completes within the accepted
      interactive time.
- [ ] A browser refresh or new session does not retain previously uploaded files.
- [ ] The public, no-sign-in access choice is approved for the information being
      uploaded.
- [ ] The administrative owner records approval for production.

## Production cutover — separate change

Do not include the following work in the review-release pull request:

- [ ] Record the production repository, branch, main-file path, access setting,
      and Python version from Streamlit.
- [ ] Tag the accepted legacy release before changing the launcher.
- [ ] Change root `app.py` to launch v2 while keeping the existing Streamlit
      deployment coordinates.
- [ ] Align root `requirements.txt` with the accepted v2 dependencies.
- [ ] Run the automated suite and hosted startup check again.
- [ ] Merge during a low-use period and smoke-test the existing production URL.

## Rollback

- [ ] Revert only the production-cutover change if the hosted build, imports,
      scheduling, exports, performance, or access setting fails acceptance.
- [ ] Confirm the legacy application is restored at the production URL.
- [ ] Keep the v2 review app available while the issue is investigated.
- [ ] Retain the legacy release through at least one successful v2 production
      campaign.
