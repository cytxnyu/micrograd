import time
import random
import numpy as np
from micrograd.engine import Value
from micrograd.nn import MLP

# Create a dataset with 100 data points
np.random.seed(42)
data = []
for _ in range(100):
    # Create random input vector of size 3
    x = np.random.randn(3).tolist()
    # Compute label based on a simple rule
    y = 1.0 if (2*x[0] + 3*x[1] - x[2]) > 0 else -1.0
    data.append((x, y))

# Initialize MLP
mlp = MLP(3, [4, 4, 1])
print("MLP:", mlp)
print("Number of parameters:", len(mlp.parameters()))

# Training parameters
learning_rate = 0.01
epochs = 100

# Test 1: Batch size = 1 (stochastic gradient descent)
print("\n--- Training with batch_size=1 ---")
start_time = time.time()

for epoch in range(epochs):
    total_loss = 0.0
    for x, y in data:
        # Forward pass
        y_pred = mlp(x)
        # Compute loss (mean squared error)
        loss = (y_pred - y) ** 2
        total_loss += loss.data
        # Backward pass
        mlp.zero_grad()
        loss.backward()
        # Update parameters
        for p in mlp.parameters():
            p.data -= learning_rate * p.grad
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/len(data):.4f}")

time_batch1 = time.time() - start_time
print(f"Training time: {time_batch1:.4f} seconds")

# Test 2: Batch size = 4
print("\n--- Training with batch_size=4 ---")
# Reinitialize MLP for fair comparison
mlp2 = MLP(3, [4, 4, 1])
start_time = time.time()

for epoch in range(epochs):
    total_loss = 0.0
    # Create batches
    random.shuffle(data)
    # Process batches without per-sample Python loops
    for i in range(0, len(data), 4):
        batch = data[i:i+4]
        # Prepare batch input
        batch_x = [x for x, y in batch]
        batch_y = [y for x, y in batch]
        # Forward pass (MLP already handles batch input)
        y_preds = mlp2(batch_x)
        # Compute loss for each sample in batch and accumulate
        batch_loss = Value(0)
        for y_pred, y in zip(y_preds, batch_y):
            batch_loss += (y_pred - y) ** 2
        avg_loss = batch_loss / len(batch)
        total_loss += avg_loss.data
        # Backward pass
        mlp2.zero_grad()
        avg_loss.backward()
        # Update parameters
        for p in mlp2.parameters():
            p.data -= learning_rate * p.grad
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/(len(data)/4):.4f}")

time_batch4 = time.time() - start_time
print(f"Training time: {time_batch4:.4f} seconds")

# Test 3: Batch size = 32
print("\n--- Training with batch_size=32 ---")
# Reinitialize MLP for fair comparison
mlp3 = MLP(3, [4, 4, 1])
start_time = time.time()

for epoch in range(epochs):
    total_loss = 0.0
    # Create batches
    random.shuffle(data)
    for i in range(0, len(data), 32):
        batch = data[i:i+32]
        # Prepare batch input
        batch_x = [x for x, y in batch]
        batch_y = [y for x, y in batch]
        # Forward pass
        y_preds = mlp3(batch_x)
        # Compute loss for each sample in batch
        losses = [(y_pred - y) ** 2 for y_pred, y in zip(y_preds, batch_y)]
        # Compute average loss
        avg_loss = sum(losses, Value(0)) / len(losses)
        total_loss += avg_loss.data
        # Backward pass
        mlp3.zero_grad()
        avg_loss.backward()
        # Update parameters
        for p in mlp3.parameters():
            p.data -= learning_rate * p.grad
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/(len(data)/32):.4f}")

time_batch32 = time.time() - start_time
print(f"Training time: {time_batch32:.4f} seconds")

# Compare training times
print("\n--- Training Time Comparison ---")
print(f"batch_size=1: {time_batch1:.4f} seconds")
print(f"batch_size=4: {time_batch4:.4f} seconds")
print(f"batch_size=32: {time_batch32:.4f} seconds")
print(f"batch_size=4 is {time_batch1/time_batch4:.2f}x faster than batch_size=1")
print(f"batch_size=32 is {time_batch1/time_batch32:.2f}x faster than batch_size=1")

# Test the trained models on the data
print("\n--- Testing Trained Models ---")
print("Model with batch_size=1:")
for x, y in data:
    y_pred = mlp(x)
    print(f"Input: {x}, True: {y}, Pred: {y_pred.data:.4f}")

print("\nModel with batch_size=4:")
for x, y in data:
    y_pred = mlp2(x)
    print(f"Input: {x}, True: {y}, Pred: {y_pred.data:.4f}")

print("\nModel with batch_size=32:")
for x, y in data[:10]:  # Test on first 10 samples
    y_pred = mlp3(x)
    print(f"Input: {x}, True: {y}, Pred: {y_pred.data:.4f}")