# ControlledResponseManager
#
# Centralises all pre-written safe responses used when the pipeline refuses or redirects.
# Response types:
#   - urgency_redirect       : direct to emergency services / medical team
#   - refuse_medical_advice  : explain the system cannot give personalized advice
#   - ask_reformulation      : invite the user to rephrase
#   - no_source_found        : no relevant document found in the knowledge base
#   - out_of_scope           : request outside the system's perimeter
#
# All messages must:
#   - Be written in plain, empathetic French
#   - Never contain medical advice
#   - Point the user toward a healthcare professional when appropriate
