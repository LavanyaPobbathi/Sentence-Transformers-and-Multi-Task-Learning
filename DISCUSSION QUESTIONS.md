# Discussion Questions

## 1. Training Strategy and Component Freezing

Based on my implementation and test results (e.g., sentiment confidence ~42.11%, subjectivity confidence ~58.26%), here's when to freeze different components:

### When to Freeze Transformer Backbone:
1. When fine-tuning for Fetch-specific terminology
   - The base BERT model already understands general English
   - Only need to adapt to terms like "points", "receipts", "rewards"
2. When data is limited (common in specialized domains)
   - Prevents overfitting on small Fetch-specific dataset
   - Maintains general language understanding
3. When computational resources are limited
   - Significantly reduces training time
   - Requires less memory

### When to Freeze One Head While Training Other:
1. When tasks have different data availability
   - E.g., if we have more sentiment-labeled data than subjectivity
   - Train sentiment head while keeping subjectivity frozen
2. When one task performs better
   - Our tests show subjectivity (58.26%) performs better than sentiment (42.11%)
   - Could freeze subjectivity head and focus on improving sentiment
3. When adding new tasks
   - Keep existing well-performing heads frozen
   - Train only the new task head

## 2. Multi-Task vs. Separate Models Decision

Based on the implementation and test results:

### Use Multi-Task Model When:
1. Tasks share common features
   - Both tasks analyze receipt/reward-related text
   - Share common understanding of Fetch-specific terms
2. Resources are limited
   - Single backbone saves memory/compute
   - Shared feature extraction (768-dim embeddings)
3. Tasks benefit from knowledge transfer
   - Sentiment and subjectivity are correlated
   - Shared understanding improves both tasks

### Use Separate Models When:
1. Tasks need different architectures
   - If one task needs specialized layers
   - If tasks require different token-level features
2. Tasks need separate deployment
   - Different scaling requirements
   - Different latency requirements
3. Tasks have conflicting requirements
   - Different optimization needs
   - Different regularization strategies

## 3. Handling Data Imbalance

Assuming Task A (sentiment) has abundant data while Task B (subjectivity) has limited data:

### Architectural Solutions:
1. Gradient balancing
   - Use gradient scaling between tasks
   - Prevent dominant task from overwhelming shared features
2. Task-specific regularization
   - Stronger dropout (0.1) for abundant task
   - Lower dropout for limited task

### Training Strategies:
1. Dynamic batch sizing
   - Larger batches for limited data task
   - Ensures adequate updates from limited data
2. Task sampling
   - Over-sample limited data task
   - Balance task representation in training

### Data Augmentation:
1. For limited data task
   - Use synonyms and paraphrasing
   - Generate synthetic examples
2. Validation strategy
   - Stricter validation for abundant task
   - More lenient metrics for limited task

These solutions are reflected in the implementation:
- Modular architecture allowing task-specific modifications
- Flexible training configurations
- Shared backbone for efficient knowledge transfer
