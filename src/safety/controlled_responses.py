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
            "⚠️ **Situation urgente — agissez immédiatement.**\n\n"
            "Si vous êtes **dans le service**, prévenez immédiatement l'équipe soignante "
            "ou appuyez sur la sonnette d'appel.\n\n"
            "Si vous êtes **hors de l'hôpital**, appelez le **15 (SAMU)** ou le **112**."
        )

    @staticmethod
    def get_out_of_scope() -> str:
        return (
            "Cette question sort du domaine de la réanimation — "
            "je ne suis pas en mesure d'y répondre.\n\n"
            "N'hésitez pas à me poser une question sur le fonctionnement du service, "
            "les équipements, les soins courants ou l'organisation des visites."
        )

    @staticmethod
    def get_refuse_medical_advice() -> str:
        return (
            "Je comprends que vous cherchez des réponses, et c'est tout à fait normal "
            "dans cette situation.\n\n"
            "Ce type de question — qui concerne directement l'état ou le traitement d'une "
            "personne — dépasse ce que je peux vous dire en tant qu'assistant d'information. "
            "**L'équipe médicale qui suit votre proche** est la seule à pouvoir vous répondre "
            "avec précision : n'hésitez pas à leur poser cette question lors du prochain échange."
        )

    @staticmethod
    def get_ask_reformulation() -> str:
        return (
            "Je n'ai pas bien saisi votre question — pourriez-vous la reformuler ?\n\n"
            "Par exemple, vous pouvez me demander à quoi sert un équipement, "
            "comment se déroule une procédure, ou comment fonctionne l'organisation du service."
        )

    @staticmethod
    def get_no_source_found() -> str:
        return (
            "Votre question concerne bien la réanimation, mais je n'ai pas trouvé "
            "d'information suffisamment fiable dans mes documents pour y répondre.\n\n"
            "Je vous encourage à **en parler directement avec l'équipe soignante**, "
            "qui pourra vous expliquer cela en tenant compte de la situation de votre proche."
        )

    @staticmethod
    def get_greeting() -> str:
        return (
            "Bonjour ! Je suis l'assistant d'information du service de réanimation.\n\n"
            "Je suis ici pour répondre à vos questions générales sur la réanimation : "
            "équipements, rôle des soignants, organisation du service, etc.\n\n"
            "N'hésitez pas à me poser votre question. \n\n"
            "⚠️ *Je ne peux pas fournir d'avis médical personnalisé. "
            "Pour toute situation urgente ou question concernant votre proche spécifiquement, "
            "adressez-vous directement à l'équipe soignante ou appelez le 15.*"
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
