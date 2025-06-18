import logging
from sentence_transformers import SentenceTransformer
from source.utils.device import get_device
from source.utils.paths import paths



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_embedding_model():
    model = SentenceTransformer(paths["embedding_nomic"], trust_remote_code=True)
    device = get_device()
    model = model.to(device)
    logger.info(f"Model loaded to {device}")
    return model