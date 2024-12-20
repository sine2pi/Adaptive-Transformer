## Adaptive Transformer

Dynamically Adjusted Base: -- The base parameter, which affects the frequencies used in positional encodings, will dynamically adjust based on the model’s performance during training. This can lead to better capture of positional information, especially if the data characteristics vary over time. done

Responsive Hyperparameter Auto-Tuning: -- The model will continuously fine-tune itself in response to the loss observed during training. This means it can potentially learn more effectively by automatically adjusting the positional encoding's influence on learning. done for base frequency - rope embeddings. done

Proactive Adjustments: -- Just like learning rate schedulers, adjusting base based on performance can help in avoiding overfitting or underfitting, and ensure the model remains effective across different training phases. done--

    Next for responsive tuning:
        Multi-Dimensional Adaptation --
  
      Track adaptation not just for base frequency, but for:
        - Rotation matrices
        - Attention span
        - Embedding dimensionality
        - Dropout rates
        - ??
    
    Conceptual Question:--
    - How do you measure "homeostasis" in a neural network?
      -- Create a "learning dynamics" module that observes and adjusts multiple model parameters simultaneously.?


--Learnable Frequencies and Phases: Introduce learnable frequencies and phase shifts in rotary embeddings to allow the model to adapt positional encodings dynamically. done

--Learnable Positional Scaling: Add a learnable scaling factor for positional embeddings to adjust the emphasis on positional information during training. done

--Multiple Rotation Matrices per Head: Use separate rotation matrices for each attention head to increase expressiveness and capture diverse positional patterns. wip

--Orthogonal Parameterization with Givens Rotations: Parameterize the rotation matrix using Givens rotations to enforce orthogonality without explicit re-orthogonalization. done

--Per-Layer Rotation Matrices: Implement different rotation matrices for each layer of the model to enhance representational capacity. wip

--Conditional Rotation Matrices: Generate rotation matrices conditioned on the input using a small neural network for dynamic positional relationships. wip

--Multi-Scale Rotary Embeddings: Use multiple sets of inverse frequencies to capture positional information at various scales within the same model. done

--Relative Positional Biases: Incorporate learnable relative positional biases into the attention scores to enhance positional understanding.done

--Regularization for Rotation Matrix: Add a regularization term to the loss function to encourage the rotation matrix to remain orthogonal during training. done


