"""Fine-tune any Hugging Face audio classification repo (`model: hf:<repo>@<sha>`).
The loop is shared: sakura.atelier.backends.hf_classification."""

from sakura.atelier.backends.hf_classification import AUDIO, ClassificationBackend

BACKEND = ClassificationBackend(AUDIO)
