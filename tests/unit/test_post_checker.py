# Unit tests — PostGenerationChecker
# Cases to cover:
#   - Safe general answer → accepted unchanged
#   - "Il va survivre" → rejected, replaced by controlled response
#   - "Donnez-lui ce médicament" → rejected
#   - Answer with no source reference → rejected
#   - was_modified flag set correctly
