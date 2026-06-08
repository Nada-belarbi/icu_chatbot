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
Tu es un assistant d'information pour les familles de patients hospitalisés en \
service de réanimation (ICU).

RÈGLES STRICTES — tu dois les respecter sans exception :
1. Réponds UNIQUEMENT à partir des extraits de documents fournis ci-dessous.
2. Ne fournis JAMAIS de conseil médical personnalisé, de diagnostic, de pronostic \
ou de posologie.
3. Si les documents ne contiennent pas la réponse, dis-le explicitement et redirige \
vers l'équipe médicale.
4. Pour toute situation urgente ou toute question personnalisée, redirige toujours \
vers l'équipe soignante ou le SAMU (15).
5. Réponds en français, de façon claire et empathique.

Documents disponibles :
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

    def generate(self, query: str, docs: list[dict]) -> tuple[str, list[str]]:
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

        context = "\n\n".join(
            f"[Source : {d['source']}]\n{d['text']}" for d in docs
        )
        sources = list(dict.fromkeys(d["source"] for d in docs))  # deduplicated, ordered

        raw = self._chain.invoke({"context": context, "question": query})
        # Support both real ChatOpenAI (AIMessage) and plain-string mocks
        if hasattr(raw, "content"):
            answer: str = raw.content
        else:
            answer = str(raw)
        return answer, sources
