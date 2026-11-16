"""Fine-tune any Hugging Face text classification repo (`model: hf:<repo>@<sha>`).
The loop is shared: sakura.atelier.backends.hf_classification."""

from sakura.atelier.backends.hf_classification import TEXT, ClassificationBackend

BACKEND = ClassificationBackend(TEXT)
