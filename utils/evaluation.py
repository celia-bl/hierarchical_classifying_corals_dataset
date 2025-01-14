from typing import Optional
from pydantic import BaseModel, Field

class BaseEvaluationRequest(BaseModel):
    test_size: float = Field(0.2, ge=0.0, le=1.0, description="Size of the test split (between 0 and 1)")
    test_seed: int = Field(42, ge=0, description="Random seed for reproducibility")

class TextEvaluationRequest(BaseEvaluationRequest):
    dataset_name: str = Field("QuotaClimat/frugalaichallenge-text-train",
                            description="The name of the dataset on HuggingFace Hub")

class ImageEvaluationRequest(BaseEvaluationRequest):
    dataset_name: str = Field("pyronear/pyro-sdis",
                            description="The name of the dataset on HuggingFace Hub")

class AudioEvaluationRequest(BaseEvaluationRequest):
    dataset_name: str = Field("rfcx/frugalai",
                            description="The name of the dataset on HuggingFace Hub")