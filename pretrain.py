from transformers import (BertTokenizer,
                          BertForMaskedLM,
                          DataCollatorForLanguageModeling,
                          Trainer,
                          TrainingArguments)
from datasets import load_dataset
import matplotlib.pyplot as plt

# 自定义 Trainer 以记录每一轮的 loss
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch_loss = []

    def training_step(self, model, inputs):
        loss = super().training_step(model, inputs)
        if self.state.global_step % len(self.get_train_dataloader()) == 0:
            self.epoch_loss.append(loss.item())
        return loss

# 只用mlm预训练
def pretrain_mlm_only():
    # 加载自定义的语料库
    datasets = load_dataset('text', data_files={'train': 'data/不安全事件分类整理.txt'})

    # 选择加载初始模型
    tokenizer = BertTokenizer.from_pretrained('pretrained-models/update-toke-bert-base-chinese')
    model = BertForMaskedLM.from_pretrained('pretrained-models/update-toke-bert-base-chinese')

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

    tokenized_datasets = datasets.map(tokenize_function, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
  
    # 定义训练参数  
    training_args = TrainingArguments(  
        output_dir='pretrained-models/self-bert-pretrain',
        overwrite_output_dir=True,
        num_train_epochs=20,
        per_device_train_batch_size=16,
        save_steps=10000,
        save_total_limit=2,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets['train'],
    )

    trainer.train()

    # 保存模型
    trainer.save_model('pretrained-models/self-bert-pretrain')

    # 可视化训练过程中的 loss
    plt.plot(trainer.epoch_loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.show()


if __name__ == "__main__":
    pretrain_mlm_only()