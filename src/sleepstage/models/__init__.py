# Register all models on import.
from sleepstage.models.attnsleep import AttnSleep
from sleepstage.models.cnn1d import CNN1D
from sleepstage.models.cnn_bilstm import CNNBiLSTM

__all__ = ["CNN1D", "CNNBiLSTM", "AttnSleep", "MODEL_REGISTRY"]

from sleepstage.models.base import MODEL_REGISTRY
