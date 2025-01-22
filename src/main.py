import torch
from transformers import AutoTokenizer
from src.models.multi_task_transformer import MultiTaskTransformer
import logging
from typing import List, Dict, Tuple
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer with models and tokenizer."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        logger.info("Tokenizer loaded successfully")
        
        # Initialize model
        logger.info("Loading model...")
        self.model = MultiTaskTransformer(
            base_model="bert-base-uncased",
            num_classes_a=3,  # Positive, Negative, Neutral
            num_classes_b=2   # Subjective/Objective
        ).to(self.device)
        logger.info("Model loaded successfully")
        
        # Define label mappings
        self.sentiment_labels = ["Negative", "Neutral", "Positive"]
        self.subjectivity_labels = ["Objective", "Subjective"]

    def predict(self, sentences: List[str]) -> List[Dict[str, dict]]:
        """
        Predict sentiment and subjectivity for a list of sentences.
        
        Args:
            sentences: List of input sentences
            
        Returns:
            List of predictions with confidence scores
        """
        # Tokenize inputs
        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=128
        ).to(self.device)
        
        # Get model predictions
        with torch.no_grad():
            task_a_out, task_b_out = self.model(
                inputs["input_ids"],
                inputs["attention_mask"]
            )
            
        # Get probabilities
        sentiment_probs = torch.softmax(task_a_out, dim=1)
        subjectivity_probs = torch.softmax(task_b_out, dim=1)
        
        # Process predictions
        results = []
        for idx, sentence in enumerate(sentences):
            # Get sentiment prediction
            sentiment_idx = torch.argmax(sentiment_probs[idx]).item()
            sentiment_conf = sentiment_probs[idx][sentiment_idx].item()
            
            # Get subjectivity prediction
            subj_idx = torch.argmax(subjectivity_probs[idx]).item()
            subj_conf = subjectivity_probs[idx][subj_idx].item()
            
            results.append({
                "sentence": sentence,
                "sentiment": {
                    "label": self.sentiment_labels[sentiment_idx],
                    "confidence": sentiment_conf * 100
                },
                "subjectivity": {
                    "label": self.subjectivity_labels[subj_idx],
                    "confidence": subj_conf * 100
                }
            })
        
        return results

def main():
    """Main function to demonstrate model capabilities."""
    logger.info("Starting script...")
    
    # Initialize analyzer
    analyzer = SentimentAnalyzer()
    
    # Test sentences
    test_sentences = [
        "The Fetch app rewards are amazing, I earned 5000 points!",
        "Receipt uploaded successfully on 2024-01-20.",
        "I'm disappointed with the delayed reward points credit.",
        "My gift card balance is $25.50.",
        "This app has completely transformed how I shop!",
        "The receipt scan failed to process."
    ]
    
    logger.info("\nProcessing sample sentences:")
    for sentence in test_sentences:
        logger.info(f"\nInput sentence: {sentence}")
        
    logger.info("\nGenerating predictions...")
    results = analyzer.predict(test_sentences)
    
    logger.info("\nModel Outputs:")
    for result in results:
        logger.info(f"\nSentence: {result['sentence']}")
        logger.info(f"Task A (Sentiment) - Prediction: {result['sentiment']['label']} "
                   f"(Confidence: {result['sentiment']['confidence']:.2f}%)")
        logger.info(f"Task B (Subjectivity) - Prediction: {result['subjectivity']['label']} "
                   f"(Confidence: {result['subjectivity']['confidence']:.2f}%)")
    
    # Test embedding visualization
    inputs = analyzer.tokenizer(
        test_sentences,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(analyzer.device)
    
    with torch.no_grad():
        embeddings = analyzer.model.backbone(
            inputs["input_ids"],
            inputs["attention_mask"]
        )
    
    logger.info(f"\nEmbedding shape for each sentence: {embeddings.shape}")
    logger.info(f"Sample embedding values for first sentence (first 5 dimensions): "
               f"{embeddings[0, :5]}")

if __name__ == "__main__":
    main()