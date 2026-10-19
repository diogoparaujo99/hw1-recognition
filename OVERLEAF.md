# Connect this repository to Overleaf

GitHub synchronization is available on your Overleaf subscription. The connection must be authorized in your account; no Overleaf account connector is available in this workspace.

Before importing, publish the completed local files to GitHub. Review the changes, then run:

```bash
cd /home/daraujo/courses/vlr/hw1-recognition
git diff --stat
git add .
git commit -m "Complete VLR homework implementations and report"
git push origin main
```

This initial push is needed because cloning and editing locally do not change the GitHub repository.

1. Open [Overleaf projects](https://www.overleaf.com/project).
2. Choose **New Project → Import from GitHub**. If prompted, connect the GitHub account that owns `diogoparaujo99/hw1-recognition` and authorize access to that repository.
3. Select `hw1-recognition` and import it as a new project.
4. Set the main document to the top-level **main.tex** and use the **pdfLaTeX** compiler. It loads the supplied template from `report/`; no extra LaTeX packages were added.
5. Click **Recompile** and check the complete report. Complete the collaboration survey in `report/main.tex` before final submission.
6. For future local edits, commit and push to the linked GitHub branch, then open **GitHub** under the Overleaf project's synchronization options and pull the changes. Synchronization is explicit, not automatic. Before editing locally again, push any Overleaf edits to GitHub and pull them locally.

```bash
cd /home/daraujo/courses/vlr/hw1-recognition
git pull --ff-only
# Edit report/main.tex or regenerate report/figures/ locally.
git add report main.tex
git commit -m "Update homework report"
git push
```

The repository contains source and report assets only. Datasets, model checkpoints, and TensorBoard logs are outside it and will not be imported into Overleaf.

If import access is unavailable, upload a ZIP of the LaTeX sources and figures as a new project. That is a manual workflow and does not establish Git synchronization.

Official instructions: [GitHub synchronization](https://docs.overleaf.com/integrations-and-add-ons/git-integration-and-github-synchronization/github-synchronization).
