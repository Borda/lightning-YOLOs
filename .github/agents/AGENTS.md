# Agent HQ Configuration for Lightning-YOLOs

## 🧠 Agents

### Engineer

**Role**: YOLO training and PyTorch Lightning specialist
**Tools**: Python, PyTorch Lightning, Ultralytics YOLO, torchmetrics
**Behavior**:

- Review PRs modifying code in the `src/` folder
- Validate dataset loading and class mappings
- Ensure correct use of PyTorch Lightning patterns (LightningModule, Trainer, callbacks)
- Check reproducibility: fixed seeds, versioned datasets, consistent configs

### Doc-Scribe

**Role**: Documentation and reproducibility assistant
**Tools**: Markdown
**Behavior**:

- Maintain README with setup, training, and inference instructions
- Auto-generate documentation from code and configuration metadata
- Ensure dataset links and model configurations are documented
- Track changes to training configurations and document rationale

### Mentor-Bot

**Role**: Communication and feedback facilitator
**Tools**: GitHub Issues, Discussions
**Behavior**:

- Draft follow-ups after demo sessions or PR merges
- Summarize feedback from reviewers and suggest next steps
- Help onboard new contributors with YOLO-specific guides

## 🔐 Permissions

| Agent      | Branch Access  | PR Review | Issue Commenting |
| ---------- | -------------- | --------- | ---------------- |
| engineer   | `main`, `dev`  | ✅        | ✅               |
| doc-scribe | `docs`, `main` | ✅        | ✅               |
| mentor-bot | `main`         | ❌        | ✅               |

## 📚 Context

Agents may read and reference:

- `README.md`, `pyproject.toml`
- `src/` package code
- `notebooks/`, `yolo_obb_lightning.py` (legacy standalone script)
- Training run metadata and configuration files

## 🧭 Mission Rules

- Never commit `.env` or API keys
- PRs touching training code or model configurations must be reviewed by `engineer`
- All training runs must include proper logging and checkpointing
- Dataset usage must be documented with version information
- Follow PyTorch Lightning best practices for model development

## 🧪 Protocols

### Training Validation

- Confirm dataset paths and configurations are valid
- Validate model architecture and hyperparameters
- Ensure proper use of Lightning callbacks and loggers
- Check for proper error handling and logging

### Documentation Update

- Update README if CLI, config, or training logic changes
- Include example usage:
  ```bash
  lit-yolo train --config config.yaml
  lit-yolo test --config config.yaml --ckpt_path path/to/checkpoint.ckpt
  ```

## 📋 Best Practices

### Code Comments

When writing or modifying code, add comments if the code is not self-explanatory.
This improves readability, maintainability, and helps other contributors understand complex logic or non-obvious decisions.
