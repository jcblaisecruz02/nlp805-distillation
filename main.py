from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer, AutoConfig
from datasets import load_dataset, load_from_disk
import torch
import torch.nn as nn
import torch.nn.functional as F

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained('google-bert/bert-base-multilingual-cased')
tokenizer.add_special_tokens({'pad_token': '[PAD]', 'eos_token': '[EOS]'})

if True:

    # Load dataset
    force_split = True

    dataset = load_dataset("Salesforce/wikitext", "wikitext-103-v1")
    if force_split or 'test' not in dataset.keys():
        split = dataset['train'].train_test_split(test_size=0.2)
        dataset['train'] = split['train']
        dataset['test'] = split['test']

    # Tokenize and preprocess the dataset
    def preprocess_function(ex):
        return tokenizer([" ".join(x) for x in ex['text']])

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=8,
        remove_columns=dataset["train"].column_names,
    )

    block_size = 512

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of block_size.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        return result

    lm_dataset = tokenized_dataset.map(group_texts, batched=True, num_proc=8)

    # Save to disk
    lm_dataset.save_to_disk('/home/jan.cruz/workspace/distillation/wikitext-102-processed')
else:
    # Load from disk
    lm_dataset = load_from_disk('/home/jan.cruz/workspace/distillation/wikitext-102-processed')

# Setup data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.alpha = alpha
        self.temperature = temperature

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        
        # place teacher on same device as student
        self._move_model_to_device(self.teacher, self.model.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):
        # compute student output
        outputs_student = model(**inputs)
        student_loss = outputs_student.loss
        
        # compute teacher output
        with torch.no_grad():
          outputs_teacher = self.teacher(**inputs)
        
        # assert size
        assert outputs_student.logits.size() == outputs_teacher.logits.size()
        
        # Soften probabilities and compute distillation loss
        loss_function = nn.KLDivLoss(reduction="batchmean")
        loss_logits = (loss_function(
            F.log_softmax(outputs_student.logits / self.args.temperature, dim=-1),
            F.softmax(outputs_teacher.logits / self.args.temperature, dim=-1)) * (self.args.temperature ** 2))
        # Return weighted student loss
        loss = self.args.alpha * student_loss + (1. - self.args.alpha) * loss_logits
        return (loss, outputs_student) if return_outputs else loss

output_dir = '/home/jan.cruz/workspace/trained-checkpoints/bert-base-multilingual-cased-distill-wikitext'

training_args = DistillationTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=16,
    fp16=True,
    learning_rate=5e-5,
    lr_scheduler_type='linear',
    warmup_ratio=0.1,
    seed=42,
    # logging & evaluation strategies
    do_eval=True,
    eval_strategy='steps',
    eval_steps=50,
    logging_dir=f"{output_dir}/logs",
    logging_strategy="steps", # to get more information to TB",
    save_strategy="steps",
    save_total_limit=2,
    load_best_model_at_end=True,
    # push to hub parameters
    push_to_hub=False,
    report_to="none",
    # distilation parameters
    alpha=0.5,
    temperature=4.0,
    full_determinism=False
    )

# Load the teacher
teacher_model = AutoModelForMaskedLM.from_pretrained('google-bert/bert-base-multilingual-cased')

# Load new config and match teacher vocab size and position embeddings
config = AutoConfig.from_pretrained('distilbert/distilbert-base-multilingual-cased')
student_model = AutoModelForMaskedLM.from_config(config)

trainer = DistillationTrainer(
    student_model,
    training_args,
    teacher_model=teacher_model,
    train_dataset=lm_dataset["train"],
    eval_dataset=lm_dataset["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
)

trainer.train()