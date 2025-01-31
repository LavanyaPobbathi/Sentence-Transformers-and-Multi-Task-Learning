# Fetch Receipt Analysis Sentence Transformers and Multi Task Learning

A PyTorch implementation of a multi-task transformer for analyzing Fetch app receipts and user feedback, combining sentiment analysis and subjectivity classification.

## Project Structure
```
sentence-transformer-mtl/
├── src/
│   ├── models/
│   │   ├── __init__.py
│   │   └── multi_task_transformer.py  # Core model implementations
│   ├── __init__.py
│   └── main.py                        # Main execution script
├── tests/
│   └── test_model.py                  # Test suite
└── requirements.txt                    # Project dependencies
```

## Installation

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage Examples

### Basic Usage
```python
from transformers import AutoTokenizer
from src.models.multi_task_transformer import MultiTaskTransformer

# Initialize
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = MultiTaskTransformer(
    base_model="bert-base-uncased",
    num_classes_a=3,  # Sentiment: Positive(0), Neutral(1), Negative(2)
    num_classes_b=2   # Subjectivity: Objective(0), Subjective(1)
)

# Process text
text = "Got 5000 points for my Target receipt!"
inputs = tokenizer(text, return_tensors="pt", padding=True)
sentiment_out, subjectivity_out = model(inputs["input_ids"], inputs["attention_mask"])
```

## Model Architecture

### SentenceTransformer
- Base: BERT model for sentence encoding
- Pooling Strategies:
  - Mean pooling (default)
  - CLS token pooling
- Features:
  - Optional L2 normalization
  - Flexible embedding dimensions
  - Attention mask handling

### MultiTaskTransformer
1. Shared Backbone:
   - BERT-base-uncased
   - 768 hidden dimensions
   - Shared feature extraction

2. Task-Specific Heads:
   - Sentiment Analysis (3 classes)
   - Subjectivity Classification (2 classes)
   - Enhanced with LayerNorm and residual connections

## Performance Metrics
Based on test results:

1. Sentiment Analysis:
   - Average confidence: ~41.42%
   - Best performance on positive statements
   - Reliable neutral classification

2. Subjectivity Classification:
   - Average confidence: ~57.66%
   - Strong objective statement detection
   - Consistent performance across categories
     
### Sample performance:

![image](https://github.com/user-attachments/assets/c1fbf0cf-aa4c-456c-8983-b442e0203c57)


## Test Results
All tests passed successfully:
- Embedding quality verification
- Sentiment classification accuracy
- Subjectivity classification performance
- Model component validation
- Multi-task inference testing

### Sample Tests:

![image](https://github.com/user-attachments/assets/6df9b44b-7e3d-46b1-87bd-d88a3538ec05)

