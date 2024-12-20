class MetricsCallback(TrainerCallback):
    def __init__(self, tb_writer, tokenizer, metric, log_every_n_steps=20):
        super().__init__()
        self.tb_writer = tb_writer
        self.tokenizer = tokenizer
        self.metric = metric
        self.log_every_n_steps = log_every_n_steps
        self.predictions = None
        self.label_ids = None

    def compute_cer(self, pred_str, label_str):
        cer = 100 * self.metric.compute(predictions=pred_str, references=label_str)
        return cer

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None:
            for key, value in metrics.items():
                if key.startswith("eval_"):
                    self.tb_writer.add_scalar(key, value, state.global_step)
                    print(f"Step {state.global_step} - {key}: {value}")

        if self.predictions is not None and self.label_ids is not None:
            pred_str = self.tokenizer.batch_decode(self.predictions, skip_special_tokens=True)
            label_str = self.tokenizer.batch_decode(self.label_ids, skip_special_tokens=True)

            sample_index = 1
            self.tb_writer.add_text("Prediction", pred_str[sample_index], state.global_step)
            self.tb_writer.add_text("Label", label_str[sample_index], state.global_step)

            print(f"Step {state.global_step} - Sample Prediction: {pred_str[sample_index]}")
            print(f"Step {state.global_step} - Sample Label: {label_str[sample_index]}")

        self.predictions = None
        self.label_ids = None

def create_compute_metrics(callback_instance):
    def compute_metrics(eval_pred):
        pred_logits = eval_pred.predictions
        label_ids = eval_pred.label_ids

        if isinstance(pred_logits, tuple):
            pred_ids = pred_logits[0]
        else:
            pred_ids = pred_logits
        if pred_ids.ndim == 3:
            pred_ids = np.argmax(pred_ids, axis=-1)

        label_ids[label_ids == -100] = callback_instance.tokenizer.pad_token_id
        callback_instance.predictions = pred_ids
        callback_instance.label_ids = label_ids

        pred_str = callback_instance.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = callback_instance.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        cer = 100 * callback_instance.metric.compute(predictions=pred_str, references=label_str)

        pred_flat = pred_ids.flatten()
        labels_flat = label_ids.flatten()
        mask = labels_flat != callback_instance.tokenizer.pad_token_id

        accuracy = accuracy_score(labels_flat[mask], pred_flat[mask])
        precision = precision_score(labels_flat[mask], pred_flat[mask], average='weighted', zero_division=0)
        recall = recall_score(labels_flat[mask], pred_flat[mask], average='weighted', zero_division=0)
        f1 = f1_score(labels_flat[mask], pred_flat[mask], average='weighted', zero_division=0)

        return {
            "cer": cer,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
    return compute_metrics
