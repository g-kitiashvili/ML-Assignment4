# ML-Assignment4
# Facial Expression Recognition Challenge

## Project Overview
This repository contains my solution for the Kaggle Facial Expression Recognition Challenge. The project demonstrates practical experience with PyTorch, systematic hyperparameter tuning, and comprehensive experiment tracking using Weights & Biases.

## Dataset Description
- **Images**: 48x48 pixel grayscale facial images
- **Classes**: 7 emotion categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)
- **Training Set**: 28,709 examples
- **Public Test Set**: 3,589 examples

## Project Structure
fer-challenge/
├── README.md
├── experiment_01_BasicNN.ipynb         # Baseline: Why CNNs are better than NNs
├── experiment_02_SimpleCNN.ipynb       # Basic CNN architecture
├── experiment_03_CNN_Overfitting.ipynb # Large CNN without regularization
├── experiment_04_CNN_Regularized.ipynb # CNN with dropout + BatchNorm + L2
├── experiment_05_CNN_DataAugmentation.ipynb # Impact of data augmentation
├── experiment_06_DeepCNN.ipynb         # Very deep architecture
├── experiment_07_TinyCNN_Underfitting.ipynb # Too small model
├── experiment_08_LearningRate_Analysis.ipynb # Compare different learning rates
├── experiment_09_Optimizer_Comparison.ipynb # SGD vs Adam vs RMSprop
└── experiment_10_BatchSize_Impact.ipynb # Small vs Large batch sizes


## Experimental Approach and Decision Rationale

### Experiment 01: Basic Neural Network Baseline
**Why this experiment?**
- Establish a baseline using traditional fully connected neural networks
- Demonstrate the limitations of treating images as flat vectors
- Show why spatial information is crucial for image classification

**Architecture Decision:**
- 3 hidden layers with [512, 256, 128] neurons
- ReLU activation functions
- No convolutions to intentionally show poor performance

**Expected Outcome:**
- Low accuracy (~25%) due to loss of spatial information
- High parameter count (1.28M) with poor performance
- Clear motivation for using CNNs

**Actual Results:**
- Validation accuracy: ~26%
- Severe limitation in learning spatial patterns
- Confirms hypothesis that spatial structure is essential

---

### Experiment 02: Simple CNN
**Why this experiment?**
- Introduce convolutional layers to capture spatial features
- Establish CNN baseline for comparison
- Keep architecture simple to understand basic improvements

**Architecture Decision:**
- 2 convolutional blocks (32, 64 channels)
- Max pooling after each block
- 2 fully connected layers [128, 64]
- Minimal complexity to isolate CNN benefits

**Expected Outcome:**
- Significant improvement over basic NN
- Better feature extraction with fewer parameters
- Some overfitting without regularization

**Actual Results:**
- Validation accuracy: ~45% (19% improvement)
- 10x fewer parameters than basic NN
- Validates importance of convolutions

---

### Experiment 03: Large CNN Without Regularization
**Why this experiment?**
- Understand overfitting behavior in deep networks
- Establish upper bound of model capacity
- Motivate need for regularization techniques

**Architecture Decision:**
- 5 convolutional blocks (VGG-inspired)
- Channels: [64, 128, 256, 512, 512]
- Large FC layers [1024, 512]
- NO dropout, batch norm, or weight decay

**Expected Outcome:**
- Very high training accuracy
- Poor validation accuracy
- Large gap indicating overfitting

**Actual Results:**
- Training accuracy: ~92%
- Validation accuracy: ~38%
- Overfitting gap: 54%
- Clear demonstration of memorization vs learning

---

### Experiment 04: CNN with Regularization
**Why this experiment?**
- Combat overfitting identified in Experiment 03
- Test multiple regularization techniques together
- Find balance between capacity and generalization

**Architecture Decision:**
- 4 convolutional blocks (moderate depth)
- Batch Normalization after each conv layer
- Dropout: 0.2 after conv, 0.5 after FC
- L2 regularization (weight_decay=0.001)

**Regularization Rationale:**
- **Batch Norm**: Stabilizes training, reduces internal covariate shift
- **Dropout**: Prevents co-adaptation of neurons
- **L2**: Penalizes large weights, encourages simpler solutions

**Expected Outcome:**
- Reduced overfitting gap
- Better validation accuracy
- More stable training

**Actual Results:**
- Validation accuracy: ~55%
- Overfitting gap reduced to ~15%
- Confirms effectiveness of regularization

---

### Experiment 05: Data Augmentation
**Why this experiment?**
- Limited dataset size (28k samples) suggests augmentation could help
- Test if artificial data diversity improves generalization
- Build on regularized model from Experiment 04

**Augmentation Decision:**
- Rotation: ±10° (faces can be slightly tilted)
- Horizontal flip: 50% (faces are symmetric)
- Brightness: ±0.2 (lighting variations)
- Contrast: ±0.2 (camera quality variations)

**Design Rationale:**
- Conservative augmentations that preserve emotion
- No vertical flips (would change expression meaning)
- No extreme rotations (unrealistic for face images)

**Expected Outcome:**
- Further reduction in overfitting
- Improved validation accuracy
- Longer training time but better generalization

**Actual Results:**
- Validation accuracy: ~58%
- Best single improvement technique
- Overfitting gap: ~12%

---

### Experiment 06: Deep CNN with Residual Connections
**Why this experiment?**
- Test if very deep architectures help
- Use residual connections to enable deeper networks
- Explore diminishing returns of depth

**Architecture Decision:**
- 8 residual blocks with skip connections
- Progressive channel increase [64, 128, 256, 512]
- Global average pooling instead of large FC
- Kaiming initialization for better gradient flow

**Design Rationale:**
- Residual connections prevent vanishing gradients
- Global average pooling reduces parameters
- Deeper networks can learn more complex features

**Expected Outcome:**
- Similar or slightly better performance
- Longer training time
- Potential diminishing returns

**Actual Results:**
- Validation accuracy: ~57%
- No significant improvement over simpler models
- Training time 3x longer
- Conclusion: Depth beyond 6 layers not beneficial for this dataset

---

### Experiment 07: Tiny CNN (Underfitting Analysis)
**Why this experiment?**
- Understand lower bound of model capacity
- Identify signs of underfitting
- Establish minimum viable architecture

**Architecture Decision:**
- Single conv layer (16 channels)
- One FC layer
- Minimal parameters (~50k)

**Expected Outcome:**
- Poor performance on both train and validation
- High bias, low variance
- Clear underfitting patterns

**Actual Results:**
- Training accuracy: ~35%
- Validation accuracy: ~32%
- Model lacks capacity for complex patterns
- Confirms need for adequate model size

---

### Experiment 08: Learning Rate Analysis
**Why this experiment?**
- Learning rate is crucial hyperparameter
- Find optimal value for stable convergence
- Test learning rate scheduling

**LR Values Tested:**
- 0.1: Test aggressive learning
- 0.01: Standard starting point
- 0.001: Conservative approach
- 0.0001: Very conservative

**Scheduling Decision:**
- Cosine annealing for smooth decay
- ReduceLROnPlateau as alternative

**Expected Outcome:**
- 0.1: Training instability
- 0.001: Best balance
- 0.0001: Too slow convergence

**Actual Results:**
- 0.1: Divergence and instability
- 0.001: Optimal (best validation accuracy)
- 0.0001: 20% slower convergence
- Cosine annealing slightly better than fixed LR

---

### Experiment 09: Optimizer Comparison
**Why this experiment?**
- Different optimizers have different convergence properties
- Find best optimizer for this specific task
- Compare adaptive vs non-adaptive methods

**Optimizers Tested:**
- **SGD+Momentum**: Classic, well-understood
- **Adam**: Adaptive learning rates, fast convergence
- **RMSprop**: Adaptive, good for RNNs/CNNs
- **AdamW**: Adam with decoupled weight decay

**Expected Outcome:**
- Adam: Fastest convergence
- SGD: More stable, potentially better final result
- AdamW: Best of both worlds

**Actual Results:**
- Adam: Best overall (58% val acc)
- AdamW: Close second (57% val acc)
- SGD: Slower but stable (55% val acc)
- RMSprop: Middle ground (56% val acc)

---

### Experiment 10: Batch Size Impact
**Why this experiment?**
- Batch size affects gradient noise and generalization
- Find optimal trade-off between speed and performance
- Test batch size scaling effects

**Batch Sizes Tested:**
- 16: High gradient noise, slow
- 32: Moderate noise
- 64: Standard choice
- 128: Less noise, faster
- 256: Minimal noise, very fast

**Expected Outcome:**
- Small batches: Better generalization, slower
- Large batches: Faster training, potential generalization loss

**Actual Results:**
- 16: Best generalization (59% val acc) but 4x slower
- 64: Best trade-off (58% val acc)
- 256: Fastest but lower accuracy (54% val acc)
- Confirms noise helps generalization

---

## Key Insights and Conclusions

### Architecture Evolution
1. **Spatial Features are Critical**: Basic NN (26%) vs Simple CNN (45%) shows 19% improvement
2. **Regularization is Essential**: Reduces overfitting gap from 54% to 15%
3. **Data Augmentation Provides Largest Gain**: Single best improvement technique
4. **Depth Has Diminishing Returns**: Beyond 6 layers shows no benefit for 48x48 images
5. **Model Capacity Must Match Task**: Too small underfits, too large overfits

### Hyperparameter Insights
1. **Learning Rate**: 0.001 optimal with cosine annealing
2. **Optimizer**: Adam provides best results for this task
3. **Batch Size**: 64 balances speed and performance
4. **Dropout Rates**: 0.2 for conv, 0.5 for FC optimal
5. **Weight Decay**: 0.001 helps without over-regularizing

### Best Model Configuration
- **Architecture**: 4 Conv blocks with BatchNorm and Dropout
- **Optimizer**: Adam (lr=0.001) with cosine annealing
- **Batch Size**: 64
- **Data Augmentation**: Rotation, flip, brightness adjustments
- **Regularization**: Dropout(0.5), BatchNorm, weight_decay=0.001
- **Final Validation Accuracy**: ~62%

## Iterative Improvement Process
1. Started with baseline to understand problem complexity
2. Introduced CNNs to capture spatial features
3. Identified overfitting as main challenge
4. Applied regularization techniques systematically
5. Added data augmentation for diversity
6. Explored architecture limits (depth)
7. Fine-tuned hyperparameters
8. Combined best practices for final model

## W&B Integration
All experiments are logged to Weights & Biases with:
- Training/Validation metrics per epoch
- Model architectures and parameters
- Hyperparameter configurations
- Confusion matrices and per-class metrics
- Learning curves and overfitting analysis

## Lessons Learned
1. **Start Simple**: Baseline models reveal problem characteristics
2. **One Change at a Time**: Isolate effects of each modification
3. **Monitor Overfitting**: Gap between train/val is key metric
4. **Data Quality > Model Complexity**: Augmentation more effective than depth
5. **Systematic Experimentation**: Document reasoning for reproducibility

## Future Improvements
- Transfer learning with pre-trained models
- Ensemble methods combining best models
- Advanced augmentation (MixUp, CutMix)
- Class imbalance handling
- Cross-validation for more robust results

## How to Run
1. Clone repository
2. Upload notebooks to Google Colab
3. Place kaggle.json in Google Drive
4. Run notebooks in numerical order
5. View results on W&B dashboard
