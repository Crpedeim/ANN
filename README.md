# ANN
A simple neural network I built for my blog


Here are the things I learned 


# =============================================================================
# NEURAL NETWORK MATH & GENERALIZATIONS
# =============================================================================

# --- 1. NOTATION ---
# L       : Current layer index
# A[L]    : Activation (output) of layer L. (Input is A[0])
# Z[L]    : Pre-activation (linear input) of layer L -> Z[L] = W[L] @ A[L-1] + B[L]
# W[L]    : Weight matrix connecting layer L-1 to L
# m       : Batch size (number of images processed at once)

# --- 2. FORWARD PROPAGATION ---
# The general formula for any layer L:
# Z[L] = W[L] @ A[L-1] + B[L]
# A[L] = activation_function(Z[L])  <-- e.g., ReLU for hidden, Softmax for output

# --- 3. BACKPROPAGATION (THE TWO GOLDEN RULES) ---

# RULE A: Calculating the "Error Term" (Delta) for any Hidden Layer
# We pull the error from the NEXT layer (L+1) backwards to the current layer (L).
# Formula: Delta[L] = (W[L+1].T @ Delta[L+1]) * derivative_of_activation(Z[L])
# Logic:   "Distribute the future error back to where it came from,
#           then switch off neurons that weren't active (derivative term)."

# RULE B: Calculating the Gradient for Weights (Delta_W)
# To find how much to change weights, we combine the layer's error with its input.
# Formula: Gradient_W[L] = (1/m) * (Delta[L] @ A[L-1].T)
# Logic:   "Error * Input". If Input was high AND Error was high,
#           this weight needs a big change.

# --- 4. LOSS FUNCTIONS & SPECIAL CASES ---

# Case 1: Output Layer with Softmax + Categorical Cross-Entropy
# The complex derivatives cancel out perfectly to give a simple subtraction.
# Delta_Output = A[Output] - Y_Target
# (Where Y_Target is the one-hot encoded ground truth)


# --- 5. WHY RELU & CROSS-ENTROPY WORK: VANISHING GRADIENTS ---
#
# Phenomenon: Switching from Sigmoid/MSE to ReLU/Cross-Entropy sped up learning.
# Problem:    "Vanishing Gradient Problem". Sigmoid derivatives are small (<0.25).
#             Multiplying small numbers in deep networks causes gradients to approach zero.
# Solution:
#   - ReLU: Derivative is either 0 or 1. Gradients flow through the network
#     without shrinking, solving the vanishing gradient problem.
#   - Cross-Entropy + Softmax: Creates a convex-like loss surface with steep gradients
#     when predictions are wrong, forcing faster correction than MSE.


# --- 6. WHY INCREASING SIZE WORKS: MODEL CAPACITY & UNIVERSAL APPROXIMATION ---
#
# Phenomenon: Increasing hidden neurons from 20 -> 128 improved accuracy (90% -> 96.7%).
# Concept:    "Model Capacity" or "Representational Power".
# Theorem:    The "Universal Approximation Theorem".
#
# Explanation:
#   - A neural network with a single hidden layer can approximate ANY continuous
#     function to arbitrary precision, *provided* it has enough neurons.
#   - With 20 neurons, the network was "Underfitting". It lacked the memory/capacity
#     to learn the complex shapes of all 10 digits simultaneously.
#   - With 128 neurons, we increased the "Hypothesis Space", allowing the network
#     to learn distinct features (loops, lines, curves) without interference.

# Case 2: Standard Gradient Descent vs. Mini-Batch
# If using Mini-Batch (m > 1):
#   - Z and A become matrices of shape (Neurons, m)
#   - We MUST divide the weight gradients by 'm' to keep updates stable.
#   - Bias update: Gradient_B[L] = (1/m) * np.sum(Delta[L], axis=1, keepdims=True)

# =============================================================================
