# Publish chatbot-plugin-sdk to PyPI

## Prerequisites

1. PyPI account: https://pypi.org/account/register/
2. API token: https://pypi.org/manage/account/#api-tokens
   - Create token with scope `Entire account` (or scoped to `chatbot-plugin-sdk` if already reserved)
3. Install build tools:
   ```bash
   pip install build twine
   ```

## Steps

```bash
cd /home/joy-teng/VSCode/chatbot-plugin-sdk

# 1. Clean old builds
rm -rf dist/ *.egg-info/

# 2. Build wheel + sdist
python -m build

# 3. Verify
ls dist/
# should show:
# chatbot_plugin_sdk-0.3.0-py3-none-any.whl
# chatbot_plugin_sdk-0.3.0.tar.gz

# 4. Upload to PyPI
python -m twine upload dist/*
# Username: __token__
# Password: <paste your PyPI API token>

# 5. Verify install
pip install chatbot-plugin-sdk
python -c "from chatbot_plugin_sdk import RagArticleProcessor, RagQueryProcessor; print('OK')"
```

## Version bump for next release

Edit `pyproject.toml`:
```toml
version = "0.3.1"  # bump me
```

Then repeat steps 1-4.
