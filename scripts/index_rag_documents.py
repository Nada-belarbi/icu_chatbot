"""
scripts/index_rag_documents.py

Indexer tous les documents RAG de data/rag_documents/raw/ dans la base Chroma.
Lance ce script une fois (ou après chaque ajout de documents).

Stratégie anti-corruption :
  - Construction dans un répertoire temporaire (chroma_db_rebuild/)
  - Swap atomique une fois terminé : chroma_db_rebuild → chroma_db
  - L'ancienne base n'est jamais supprimée tant que Streamlit l'a ouverte

Usage:
    python scripts/index_rag_documents.py
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
_log = logging.getLogger(__name__)


def _try_rmtree(path: Path) -> bool:
    """Try to delete a directory. Return True on success, False if locked."""
    try:
        shutil.rmtree(path)
        return True
    except PermissionError:
        return False


def main() -> None:
    from config.settings import CHROMA_DB_PATH, RAG_DOCUMENTS_PATH, CHROMA_COLLECTION_NAME
    from src.rag.document_loader import DocumentLoader
    from src.rag.vector_store import VectorStoreManager

    # Répertoire temporaire de reconstruction (jamais ouvert par Streamlit)
    rebuild_path = CHROMA_DB_PATH.parent / (CHROMA_DB_PATH.name + "_rebuild")
    old_path = CHROMA_DB_PATH.parent / (CHROMA_DB_PATH.name + "_old")

    _log.info("=== Indexation RAG ===")
    _log.info("Source  : %s", RAG_DOCUMENTS_PATH)
    _log.info("Cible   : %s", CHROMA_DB_PATH)

    # 1. Chargement et découpage de tous les documents
    _log.info("Chargement des documents...")
    loader = DocumentLoader()
    chunks = loader.load()
    _log.info("Total chunks générés : %d", len(chunks))

    if not chunks:
        _log.error("Aucun chunk chargé. Vérifiez le chemin source.")
        sys.exit(1)

    # Résumé par organisation
    from collections import Counter
    orgs: Counter = Counter()
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        org = src.split("/")[0] if "/" in src else "root"
        orgs[org] += 1
    _log.info("Chunks par organisation :")
    for org, count in sorted(orgs.items()):
        _log.info("  %-20s %d chunks", org, count)

    # 2. Nettoyer les restes d'une réindexation précédente
    if rebuild_path.exists():
        _log.info("Nettoyage du répertoire temporaire précédent...")
        _try_rmtree(rebuild_path)

    # 3. Construire la nouvelle base dans rebuild/
    _log.info("Indexation dans Chroma (embedding local) → %s ...", rebuild_path.name)
    vsm = VectorStoreManager(chroma_path=rebuild_path)
    vsm.index_documents(chunks)

    # 4. Swap atomique : chroma_db → chroma_db_old, chroma_db_rebuild → chroma_db
    _log.info("Remplacement de la base active...")

    if CHROMA_DB_PATH.exists():
        # Essayer de supprimer directement (si Streamlit est arrêté)
        if not _try_rmtree(CHROMA_DB_PATH):
            # Streamlit a le verrou : renommer en _old (ne supprime pas les fichiers)
            if old_path.exists():
                _try_rmtree(old_path)
            try:
                CHROMA_DB_PATH.rename(old_path)
                _log.info(
                    "Ancienne base renommée → %s (sera supprimée au prochain lancement).",
                    old_path.name,
                )
            except Exception as e:
                _log.error(
                    "Impossible de remplacer la base active : %s\n"
                    "Arrêtez Streamlit puis réexécutez ce script.",
                    e,
                )
                sys.exit(1)

    rebuild_path.rename(CHROMA_DB_PATH)

    # 5. Nettoyer l'éventuelle ancienne base renommée
    if old_path.exists():
        if _try_rmtree(old_path):
            _log.info("Ancienne base supprimée.")
        else:
            _log.warning(
                "L'ancienne base (%s) n'a pas pu être supprimée (fichiers verrouillés). "
                "Elle sera ignorée et peut être supprimée manuellement plus tard.",
                old_path,
            )

    _log.info("=== Indexation terminée — base active : %s ===", CHROMA_DB_PATH)


if __name__ == "__main__":
    main()
