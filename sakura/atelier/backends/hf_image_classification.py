"""Fine-tune any Hugging Face image classification repo (`model: hf:<repo>@<sha>`).
The loop is shared: sakura.atelier.backends.hf_classification."""

from sakura.atelier.backends.hf_classification import IMAGE, ClassificationBackend

BACKEND = ClassificationBackend(IMAGE)
