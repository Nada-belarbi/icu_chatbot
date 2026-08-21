# RAGChain
#
# Responsibilities:
#   - Build the LangChain chain: retrieved docs → prompt → OpenAI LLM → response
#   - Inject guardrail system prompt (no diagnosis, no treatment, source-only answers)
#   - Attach source references to the returned response
#   - Only called after DecisionEngine returns allow_rag
#   - Output: (answer: str, sources: list[str])
#
# Security note: only send anonymized query + document excerpts to OpenAI — never PII

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate

_log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strict system prompt — injected into every LLM call
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
Tu es un assistant bienveillant qui aide les familles de patients hospitalisés \
en service de réanimation. Tu leur expliques le fonctionnement de la réanimation \
de façon claire, chaleureuse et accessible — comme si tu parlais à quelqu'un \
d'inquiet qui a besoin de comprendre.

COMMENT RÉPONDRE :
- Appuie-toi en priorité sur les documents fournis. S'ils abordent le sujet \
même indirectement, synthétise l'information de plusieurs sources si nécessaire.
- Pour les définitions médicales générales et factuelles (équipements courants, \
procédures standard), tu peux compléter depuis tes connaissances générales si \
les documents ne couvrent pas complètement — mais reste factuel et prudent.
- Réponds en français naturel et empathique. Évite le ton administratif ou froid.
- Explique les acronymes et termes médicaux en langage simple.
- Si la question contient "mon père", "ma mère", "mon proche", reformule ta réponse au niveau général sans reprendre ces détails personnels.
- Préfère "peut", "généralement", "dans la plupart des cas" aux affirmations absolues.
- Sois concis : 3 à 6 phrases suffisent sauf si la question l'exige.
- Si l'échange a un historique, tiens-en compte pour rendre la réponse cohérente.

LIMITES ABSOLUES — ne jamais franchir :
1. Jamais de conseil médical personnalisé, diagnostic, pronostic ni posologie.
2. Si la question glisse vers l'état d'un patient spécifique, réponds sur \
le plan général uniquement et invite à parler à l'équipe soignante.
3. Urgence médicale → redirige immédiatement vers l'équipe soignante ou le 15.

{conversation_history}

Documents de référence :
{context}
"""


class RAGChain:
    """
    LangChain LCEL chain: context + question → strict system prompt → OpenAI → answer.

    Parameters
    ----------
    llm:
        Any LangChain-compatible chat model.  Pass a mock in unit tests to
        avoid real API calls.  Defaults to ``ChatOpenAI(OPENAI_MODEL)``.
    """

    def __init__(self, llm=None) -> None:
        if llm is not None:
            self._llm = llm
        else:
            from langchain_openai import ChatOpenAI
            from config.settings import OPENAI_API_KEY, OPENAI_MODEL

            self._llm = ChatOpenAI(
                model=OPENAI_MODEL,
                api_key=OPENAI_API_KEY,
                temperature=0,
            )
        self._prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                ("human", "{question}"),
            ]
        )
        # Chain without StrOutputParser — content extracted in generate()
        # to stay compatible with both real ChatOpenAI (returns AIMessage)
        # and RunnableLambda mocks used in tests.
        self._chain = self._prompt | self._llm

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self, query: str, docs: list[dict], history: list[dict] | None = None) -> tuple[str, list[str]]:
        """
        Generate an answer grounded in *docs*.

        Parameters
        ----------
        query:
            Cleaned user question (no PII).
        docs:
            List of dicts ``{text, source, score}`` from ``Retriever.search()``.

        Returns
        -------
        (answer, sources)
            *sources* is a deduplicated list of source filenames used.

        Raises
        ------
        Exception
            Propagates any OpenAI / network error so the pipeline can catch it.
        """
        if not docs:
            return "", []

        # Format conversation history for the LLM (last 3 turns max, assistant answers truncated)
        history_text = ""
        if history:
            lines = []
            for msg in history[-6:]:  # up to 3 user+assistant pairs
                role = "Famille" if msg["role"] == "user" else "Assistant"
                content = msg["content"]
                if msg["role"] == "assistant" and len(content) > 250:
                    content = content[:250] + "…"
                lines.append(f"{role} : {content}")
            if lines:
                history_text = "Historique récent :\n" + "\n".join(lines) + "\n"

        context = "\n\n".join(
            f"[Source : {d['source']}]\n{d['text']}" for d in docs
        )
        sources = list(dict.fromkeys(d["source"] for d in docs))  # deduplicated, ordered

        raw = self._chain.invoke({
            "context": context,
            "conversation_history": history_text,
            "question": query,
        })
        # Support both real ChatOpenAI (AIMessage) and plain-string mocks
        if hasattr(raw, "content"):
            answer: str = raw.content
        else:
            answer = str(raw)

        # If the LLM generated a refusal (did not use the docs), suppress sources
        # to avoid showing misleading document references on refuse responses.
        _REFUSAL_SIGNALS = (
            "je ne peux pas",
            "je ne suis pas autorisé",
            "il m'est impossible",
            "je ne suis pas en mesure",
            "adressez-vous directement",
        )
        if any(sig in answer.lower() for sig in _REFUSAL_SIGNALS):
            return answer, []

        return answer, sources
