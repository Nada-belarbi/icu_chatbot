# AnonymizedLogger
#
# Responsibilities:
#   - Log each interaction to logs/ in JSON/CSV format
#   - Fields to log:
#       timestamp | anonymized_query_hash | predicted_class | confidence_score
#       decision | sources_used | response_type | post_check_modified
#   - NEVER log: raw user text, patient names, personal data, IP addresses
#   - Log rotation: configurable max file size / daily rotation
