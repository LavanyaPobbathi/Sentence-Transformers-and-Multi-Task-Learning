import torch
import pytest
from transformers import AutoTokenizer
from src.models.multi_task_transformer import SentenceTransformer, MultiTaskTransformer
import torch.nn.functional as F

@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("bert-base-uncased")

@pytest.fixture
def sentence_transformer():
    return SentenceTransformer(model_name="bert-base-uncased")

@pytest.fixture
def multi_task_model():
    model = MultiTaskTransformer(
        base_model="bert-base-uncased",
        num_classes_a=3,  # Positive (0), Neutral (1), Negative (2)
        num_classes_b=2   # Objective (0), Subjective (1)
    )
    return model

def test_embedding_quality(tokenizer, sentence_transformer):
    """Test that similar sentences have similar embeddings."""
    sentence_pairs = [
        (
            "Uploaded my Walmart receipt to Fetch",
            "Just scanned a Target receipt in Fetch",
            True  # Should be similar
        ),
        (
            "Got 5000 points for my receipt",
            "The weather is cloudy today",
            False  # Should not be similar
        )
    ]
    
    for sent1, sent2, should_be_similar in sentence_pairs:
        inputs1 = tokenizer(sent1, return_tensors="pt", padding=True)
        inputs2 = tokenizer(sent2, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            emb1 = sentence_transformer(inputs1["input_ids"], inputs1["attention_mask"])
            emb2 = sentence_transformer(inputs2["input_ids"], inputs2["attention_mask"])
        
        similarity = F.cosine_similarity(emb1, emb2)
        
        if should_be_similar:
            assert similarity > 0.5, f"Similar sentences should have higher similarity: {similarity}"
        else:
            assert similarity < 0.5, f"Different sentences should have lower similarity: {similarity}"

def test_sentiment_classification(tokenizer, multi_task_model):
    """Test sentiment classification performance."""
    test_cases = [
        (
            "Got bonus points from Fetch!",
            0,  # Positive
            "Positive sentiment test"
        ),
        (
            "Scanned a receipt from Target",
            1,  # Neutral
            "Neutral sentiment test"
        ),
        (
            "App crashed while uploading receipt",
            2,  # Negative
            "Negative sentiment test"
        )
    ]
    
    for text, expected_class, test_name in test_cases:
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            sentiment_out, _ = multi_task_model(inputs["input_ids"], inputs["attention_mask"])
        
        pred_class = torch.argmax(sentiment_out, dim=1)
        confidence = torch.softmax(sentiment_out, dim=1).max()
        
        print(f"\nText: {text}")
        print(f"Expected class: {expected_class}")
        print(f"Predicted class: {pred_class.item()}")
        print(f"Confidence: {confidence.item():.4f}")
        
        assert confidence > 0.33, \
            f"Very low confidence ({confidence:.2f}) for: {text}"

def test_subjectivity_classification(tokenizer, multi_task_model):
    """Test subjectivity classification performance."""
    test_cases = [
        (
            "Receipt scanned at 3:45 PM",
            0,  # Objective
            "Objective statement test"
        ),
        (
            "Love how Fetch rewards adds up!",
            1,  # Subjective
            "Subjective statement test"
        )
    ]
    
    for text, expected_class, test_name in test_cases:
        inputs = tokenizer(text, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            _, subj_out = multi_task_model(inputs["input_ids"], inputs["attention_mask"])
        
        pred_class = torch.argmax(subj_out, dim=1)
        confidence = torch.softmax(subj_out, dim=1).max()
        
        print(f"\nText: {text}")
        print(f"Expected class: {expected_class}")
        print(f"Predicted class: {pred_class.item()}")
        print(f"Confidence: {confidence.item():.4f}")
        
        assert confidence > 0.33, \
            f"Very low confidence ({confidence:.2f}) for: {text}"

def test_model_components(multi_task_model):
    """Test model component shapes and configurations."""
    assert hasattr(multi_task_model, 'backbone'), "Model should have a backbone"
    assert hasattr(multi_task_model, 'task_a_head'), "Model should have task A head"
    assert hasattr(multi_task_model, 'task_b_head'), "Model should have task B head"
    
    # Check embedding dimension
    test_input = torch.randint(0, 1000, (1, 10))
    test_mask = torch.ones(1, 10)
    
    with torch.no_grad():
        embeddings = multi_task_model.backbone(test_input, test_mask)
        assert embeddings.shape[1] == 768, "Wrong embedding dimension"
        
        task_a_out = multi_task_model.task_a_head(embeddings)
        task_b_out = multi_task_model.task_b_head(embeddings)
        
        assert task_a_out.shape[1] == 3, "Wrong number of sentiment classes"
        assert task_b_out.shape[1] == 2, "Wrong number of subjectivity classes"

def test_multi_task_performance(tokenizer, multi_task_model):
    """Test combined task performance."""
    text = "Fetch app saved me money on groceries!"
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    
    with torch.no_grad():
        sent_out, subj_out = multi_task_model(inputs["input_ids"], inputs["attention_mask"])
        
        sent_probs = torch.softmax(sent_out, dim=1)
        sent_conf = sent_probs.max()
        print(f"\nSentiment confidence: {sent_conf.item():.4f}")
        assert sent_conf > 0.33, f"Very low sentiment confidence: {sent_conf}"
        
        subj_probs = torch.softmax(subj_out, dim=1)
        subj_conf = subj_probs.max()
        print(f"Subjectivity confidence: {subj_conf.item():.4f}")
        assert subj_conf > 0.33, f"Very low subjectivity confidence: {subj_conf}"