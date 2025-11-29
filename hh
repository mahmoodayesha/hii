# =========================
# 5. Evaluate on NEGATED subset
# =========================
y_true_neg = []
y_pred_neg = []

for row in negated_subset:
    y_true_neg.append(row["label"])
    y_pred_neg.append(predict_label(row["premise"], row["hypothesis"]))

y_true_neg = np.array(y_true_neg)
y_pred_neg = np.array(y_pred_neg)

acc_negated = accuracy_score(y_true_neg, y_pred_neg)
cm_negated = confusion_matrix(y_true_neg, y_pred_neg, labels=[0, 1, 2])

print("Accuracy on naturally negated hypotheses:", acc_negated)
print("Confusion matrix (rows=true, cols=pred) for negated subset:\n", cm_negated)

labels = ["entailment", "neutral", "contradiction"]

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_negated,
    annot=True,
    fmt="d",
    cmap="Reds",
    xticklabels=labels,
    yticklabels=labels
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix: Negated Subset")
plt.show()

# Show first 10 misclassified examples
misclassified_neg = []
for i, row in enumerate(negated_subset):
    if y_true_neg[i] != y_pred_neg[i]:
        misclassified_neg.append({
            "premise": row["premise"],
            "hypothesis": row["hypothesis"],
            "true_label": int(y_true_neg[i]),
            "pred_label": int(y_pred_neg[i])
        })

print(f"Total misclassified (negated subset): {len(misclassified_neg)}\n")

for example in misclassified_neg[:10]:
    print("Premise: ", example["premise"])
    print("Hypothesis: ", example["hypothesis"])
    print("True label: ", example["true_label"])
    print("Predicted label: ", example["pred_label"])
    print("-" * 50)



# =========================
# 7. Breakdown: how often predicted as CONTRADICTION
# =========================
def breakdown_to_contradiction(y_true, y_pred, name: str):
    print(f"\n{name}: how often predicted as CONTRADICTION (label=2)")
    for true_lbl, label_name in zip([0, 1, 2], ["entailment", "neutral", "contradiction"]):
        mask = (y_true == true_lbl)
        if mask.sum() == 0:
            print(f"No examples with true label = {label_name}")
            continue
        total = mask.sum()
        pred_contra = ((y_pred == 2) & mask).sum()
        print(
            f"True {label_name}: {pred_contra}/{total} "
            f"({pred_contra / total:.2%}) predicted as contradiction"
        )

breakdown_to_contradiction(y_true_neg, y_pred_neg, "Negated subset")
breakdown_to_contradiction(y_true_non, y_pred_non, "Non-negated subset")


# =========================
# 8. (Optional) Minimal-pair probe for negation
# =========================
label_map = {0: "entailment", 1: "neutral", 2: "contradiction"}

def show_minimal_pair(premise, hyp1, hyp2):
    l1 = predict_label(premise, hyp1)
    l2 = predict_label(premise, hyp2)
    print("Premise:", premise)
    print("Hyp 1:", hyp1, "->", label_map[l1])
    print("Hyp 2:", hyp2, "->", label_map[l2])
    print("-" * 50)

show_minimal_pair(
    "A woman is laughing.",
    "The woman is laughing.",
    "The woman is not laughing."
)

show_minimal_pair(
    "A man is riding a bike.",
    "The man is riding a bike.",
    "The man is not riding a bike."
)
