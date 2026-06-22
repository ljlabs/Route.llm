# Git Rules

## Never Force-Add Ignored Files

If a file is in `.gitignore`, it is ignored **on purpose**. Never use `git add -f` to override `.gitignore`. If a file is ignored, assume it contains secrets, credentials, or local-only config that must not be committed.

## Respect the User's .gitignore

- Do not question why files are ignored.
- Do not suggest force-adding ignored files.
- If a script or manifest contains sensitive data (API keys, tokens, credentials), it belongs in `.gitignore` and should stay there.
- If you create files that contain secrets, ensure they are already covered by `.gitignore` patterns or add them to `.gitignore`.
