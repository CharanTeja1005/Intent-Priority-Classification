from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-6")

text = "I need a new laptop. It's ok if it takes time"
text = "The washroom area is unhygeinic asap"
labels = ["request", "concern", "suggestion"]

result = classifier(text, labels)
print(result)
best_label = result['labels'][0]
print(best_label)

priority_labels = ["high priority", "medium priority", "low priority"]

# Perform zero-shot classification
priority_result = classifier(text, priority_labels)

# Extract the label with the highest score for priority
priority = priority_result['labels'][0]
print(priority)