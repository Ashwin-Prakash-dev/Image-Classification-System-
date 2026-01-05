import torch

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

TEMPERATURE = 2.0
ENTROPY_THRESHOLD = 1.2

def predict(model, x):
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits / TEMPERATURE, dim=1)

        confidence, pred = probs.max(dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

    decision = "accepted"
    if entropy.item() > ENTROPY_THRESHOLD:
        decision = "rejected"

    return {
        "label": CIFAR10_CLASSES[int(pred.item())],
        "confidence": round(confidence.item(), 4),
        "entropy": round(entropy.item(), 4),
        "decision": decision
    }
