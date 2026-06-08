# ControlledResponseManager
#
# Centralises all pre-written safe responses used when the pipeline refuses or redirects.
# Response types:
#   - urgency_redirect       : direct to emergency services / medical team
#   - refuse_medical_advice  : explain the system cannot give personalized advice
#   - ask_reformulation      : invite the user to rephrase
#   - no_source_found        : no relevant document found in the knowledge base
#   - out_of_scope           : request outside the system's perimeter
#   - api_unavailable        : OpenAI API is unreachable or returned an error
#   - generic_error          : unexpected exception — last-resort safe message
#


class ControlledResponseManager:
    """
    Single source of truth for all pre-written responses.

    All messages are in French, empathetic, and contain zero medical content.
    When in doubt, the message redirects to the medical team or SAMU.
    """

    # ------------------------------------------------------------------
    # Safety / routing responses
    # ------------------------------------------------------------------

    @staticmethod
    def get_urgency_redirect() -> str:
        return (
            "⚠️ **Situation urgente détectée.**\n\n"
            "Je ne suis pas en mesure de répondre à des situations d'urgence médicale. "
            "Veuillez **contacter immédiatement l'équipe médicale** présente ou appelez le **15 (SAMU)** "
            "si vous êtes hors de l'hôpital.\n\n"
            "Votre sécurité est la priorité absolue."
        )

    @staticmethod
    def get_refuse_medical_advice() -> str:
        return (
            "Je comprends votre question, mais je ne suis pas autorisé à fournir "
            "des conseils médicaux personnalisés — diagnostic, dosage ou pronostic inclus.\n\n"
            "Pour toute question concernant votre situation spécifique, "
            "**adressez-vous directement à l'équipe soignante** qui vous prend en charge. "
            "Elle dispose de toutes les informations nécessaires pour vous répondre avec précision."
        )

    @staticmethod
    def get_ask_reformulation() -> str:
        return (
            "Je n'ai pas bien compris votre question. Pourriez-vous la reformuler "
            "de façon plus précise ?\n\n"
            "Par exemple, vous pouvez me demander des informations générales sur "
            "les soins en réanimation, les équipements ou l'organisation du service."
        )

    @staticmethod
    def get_no_source_found() -> str:
        return (
            "Je n'ai pas trouvé de document correspondant à votre question "
            "dans ma base de connaissances.\n\n"
            "Si vous souhaitez des informations sur ce sujet, je vous encourage à "
            "**en parler directement avec l'équipe médicale ou paramédicale**."
        )

    @staticmethod
    def get_out_of_scope() -> str:
        return (
            "Votre question dépasse le périmètre de ce système. "
            "Je suis uniquement conçu pour répondre à des questions générales "
            "sur les soins en réanimation.\n\n"
            "Pour toute autre demande, veuillez vous adresser à l'équipe soignante "
            "ou au secrétariat du service."
        )

    # ------------------------------------------------------------------
    # Technical failure responses
    # ------------------------------------------------------------------

    @staticmethod
    def get_api_unavailable() -> str:
        return (
            "Le service de génération de réponse est temporairement indisponible. "
            "Veuillez réessayer dans quelques instants.\n\n"
            "Si le problème persiste, contactez l'équipe soignante pour toute question urgente."
        )

    @staticmethod
    def get_generic_error() -> str:
        return (
            "Une erreur inattendue s'est produite. "
            "Veuillez réessayer ou contacter l'équipe soignante si vous avez besoin d'aide immédiate."
        )

# All messages must:
#   - Be written in plain, empathetic French
#   - Never contain medical advice
#   - Point the user toward a healthcare professional when appropriate
