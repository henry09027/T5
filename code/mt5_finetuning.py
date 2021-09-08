# Importing libraries
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import json
import click

# Importing the MT5 modules from huggingface/transformers
from transformers import MT5Tokenizer, MT5ForConditionalGeneration
from rich.table import Column, Table
from rich import box
from rich.console import Console
from opencc import OpenCC
from sklearn.model_selection import train_test_split

console=Console(record=True)

def display_df(df):
    """display dataframe in ASCII format"""

    console = Console()
    table = Table(
        Column("source_text", justify="center"),
        Column("target_text", justify="center"),
        title="Sample Data",
        pad_edge=False,
        box=box.ASCII,
    )

    for i, row in enumerate(df.values.tolist()):
        table.add_row(row[0], str(row[1]))

    console.print(table)

# training logger to log training progress
training_logger = Table(
    Column("Epoch", justify="center"),
    Column("Steps", justify="center"),
    Column("Loss", justify="center"),
    title="Training Status",
    pad_edge=False,
    box=box.ASCII,
)

    
def train(epoch, tokenizer, model, device, loader, optimizer, model_params):

  """
  Function to be called for training with the parameters passed from main function

  """
  
  model.train()
  for _,data in enumerate(loader, 0):
    #y = data[2].to(device, dtype = torch.long)
    #y_ids = y[:, :-1].contiguous()
    #lm_labels = y[:, 1:].clone().detach()
    #lm_labels[y[:, 1:] == tokenizer.pad_token_id] = -100
    lm_labels = data[2].to(device, dtype=torch.long)
    lm_labels[lm_labels[:, :] == tokenizer.pad_token_id] = -100
    target_mask = data[3].to(device, dtype=torch.long)
    ids = data[0].to(device, dtype = torch.long)
    mask = data[1].to(device, dtype = torch.long)
    #print(y_ids)
    #print(lm_labels)
    outputs = model(input_ids = ids, attention_mask = mask, decoder_attention_mask=target_mask, labels=lm_labels)
    loss = outputs[0]

    if _%100==0:
      training_logger.add_row(str(epoch), str(_), str(loss))
      console.print(training_logger)


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    
def validate(epoch, tokenizer, model, device, loader, model_params):

  """
  Function to evaluate model for predictions

  """
  model.eval()
  predictions = []
  actuals = []
  with torch.no_grad():
      for _, data in enumerate(loader, 0):

          y = data[2].to(device, dtype = torch.long)
          ids = data[0].to(device, dtype = torch.long)
          mask = data[1].to(device, dtype = torch.long)

          generated_ids = model.generate(
              input_ids = ids,
              attention_mask = mask, 
              max_length=150, 
              num_beams=2,
              repetition_penalty=2.5, 
              length_penalty=1.0, 
              early_stopping=True
              )
          preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]
          target = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)for t in y]

          if _%100==0:
              console.print(f'Completed {_}')


          predictions.extend(preds)
          actuals.extend(target)
  return predictions, actuals



def process_dataframe(dataframe):
    
    #traditional to simplified
    cc = OpenCC('tw2s')
    dataframe['external'] = dataframe['external'].astype(str)
    dataframe['internal'] = dataframe['internal'].astype(str)
    dataframe['external'] = dataframe['external'].map(cc.convert)
    dataframe['internal'] = dataframe['internal'].map(cc.convert)
    
    #format to seq-to-seq
    dataframe['external'] = 'stsb sentence1: '+dataframe['external']
    dataframe['internal'] = ' sentence2: '+dataframe['internal']
    dataframe['stsb'] = dataframe['external']+dataframe['internal']
    dataframe.drop(columns=['external','internal'], inplace=True)


    return dataframe

def calc_accuracy(df):

    condlist = [df['Generated Label']==df['Actual Label'], df['Generated Label']!=df['Actual Label']]
    choicelist = [1,0]
    df['result']=np.select(condlist, choicelist)
    return df['result'].sum()/len(df)

def MT5Trainer(train_dataframe, dev_dataframe, source_text, target_text, model_params, device, output_dir="test_finetune_models/"):
  
  """
  T5 trainer

  """

  # Set random seeds and deterministic pytorch for reproducibility
  torch.manual_seed(model_params["SEED"]) # pytorch random seed
  np.random.seed(model_params["SEED"]) # numpy random seed
  torch.backends.cudnn.deterministic = True

  # logging
  console.log(f"""[Model]: Loading {model_params["MODEL"]}...\n""")

  # tokenzier for encoding the text
  tokenizer = MT5Tokenizer.from_pretrained(model_params["MODEL"])

  # Defining the model. We are using t5-base model and added a Language model layer on top for generation of Summary. 
  # Further this model is sent to device (GPU/TPU) for using the hardware.
  model = MT5ForConditionalGeneration.from_pretrained(model_params["MODEL"])
  model = model.to(device)
  new_tokens=['stsb sentence1:', ' sentence2 ', '_0', '_1']
  num_added_tokens = tokenizer.add_tokens(new_tokens, special_tokens=True)
  console.log(f"Added {num_added_tokens} tokens")
  console.print(f"Added {num_added_tokens} tokens")

  # logging
  console.log("Reading data...\n")
  

  # Importing the raw dataset
  train_data = train_dataframe[[source_text, target_text]]
  dev_data = dev_dataframe[[source_text, target_text]]

  display_df(train_dataframe.head(2))
  display_df(dev_dataframe.head(2))
    
  train_dataset = train_data.sample(random_state = model_params["SEED"])
  val_dataset = dev_data.sample(random_state = model_params["SEED"])


  # console.print(f"FULL Dataset: {dataframe.shape}")
  console.print(f"TRAIN Dataset: {train_dataset.shape}")
  console.print(f"TEST Dataset: {val_dataset.shape}\n")
  
  max_length = model_params["MAX_SOURCE_TEXT_LENGTH"]
    
  train_inputs_ids, train_inputs_mask, train_targets_ids, train_targets_mask, dev_inputs_ids, dev_inputs_mask, dev_targets_ids, dev_targets_mask = [],[],[],[],[],[],[],[]
  for train_input in train_data.stsb:
      source = tokenizer.batch_encode_plus([train_input], max_length=max_length, pad_to_max_length=True, truncation=True, padding='max_length', return_tensors='pt')
      source_ids = source['input_ids'].squeeze()
      source_mask = source['attention_mask'].squeeze()
      train_inputs_ids.append(source_ids)
      train_inputs_mask.append(source_mask)
    
  for train_target in train_data.label:
      source = tokenizer.batch_encode_plus([train_target], max_length=2, pad_to_max_length=True, truncation=True, padding='max_length', return_tensors='pt')
      source_ids = source['input_ids'].squeeze()
      source_mask = source['attention_mask'].squeeze()
      train_targets_ids.append(source_ids)
      train_targets_mask.append(source_mask)
    
  for dev_input in dev_data.stsb:
      source = tokenizer.batch_encode_plus([dev_input], max_length=max_length, pad_to_max_length=True, truncation=True, padding='max_length', return_tensors='pt')
      source_ids = source['input_ids'].squeeze()
      source_mask = source['attention_mask'].squeeze()
      dev_inputs_ids.append(source_ids)
      dev_inputs_mask.append(source_mask)
    
  for dev_target in dev_data.label:
      source = tokenizer.batch_encode_plus([dev_target], max_length=2, pad_to_max_length=True, truncation=True, padding='max_length', return_tensors='pt')
      source_ids = source['input_ids'].squeeze()
      source_mask = source['attention_mask'].squeeze()
      dev_targets_ids.append(source_ids)
      dev_targets_mask.append(source_mask)
  
  train_inputs_ids = torch.stack(train_inputs_ids)
  train_inputs_mask = torch.stack(train_inputs_mask)
  train_targets_ids = torch.stack(train_targets_ids)
  train_targets_mask = torch.stack(train_targets_mask)

  dev_inputs_ids = torch.stack(dev_inputs_ids)
  dev_inputs_mask = torch.stack(dev_inputs_mask)
  dev_targets_ids = torch.stack(dev_targets_ids)
  dev_targets_mask = torch.stack(dev_targets_mask)

  print(len(train_inputs_ids), len(train_inputs_mask), len(train_targets_ids))

  train_dataset = TensorDataset(train_inputs_ids, train_inputs_mask, train_targets_ids, train_targets_mask)


  dev_dataset = TensorDataset(dev_inputs_ids, dev_inputs_mask, dev_targets_ids, dev_targets_mask)

  # Defining the parameters for creation of dataloaders
  train_params = {
      'batch_size': model_params["TRAIN_BATCH_SIZE"],
      'shuffle': True,
      'num_workers': 0
      }


  val_params = {
      'batch_size': model_params["VALID_BATCH_SIZE"],
      'shuffle': False,
      'num_workers': 0
      }

  # Creation of Dataloaders for testing and validation. This will be used down for training and validation stage for the model.
  training_loader = DataLoader(train_dataset, **train_params)
  val_loader = DataLoader(dev_dataset, **val_params)

  # Defining the optimizer that will be used to tune the weights of the network in the training session. 
  optimizer = torch.optim.Adam(params =  model.parameters(), lr=model_params["LEARNING_RATE"])


  # Training loop
  console.log('[Initiating Fine Tuning]...\n')

  for epoch in range(model_params["TRAIN_EPOCHS"]):
      train(epoch, tokenizer, model, device, training_loader, optimizer, model_params)
      
  console.log("[Saving Model]...\n")
  #Saving the model after training
  path = os.path.join(output_dir, "model_files")
  model.save_pretrained(path)
  tokenizer.save_pretrained(path)

  accuracy = 0
  # evaluating test dataset
  console.log("[Initiating Validation]...\n")
  for epoch in range(model_params["VAL_EPOCHS"]):
    predictions, actuals = validate(epoch, tokenizer, model, device, val_loader, model_params)
    final_df = pd.DataFrame({'Generated Label':predictions,'Actual Label':actuals})
    accuracy = calc_accuracy(final_df)
    final_df.to_csv(os.path.join(output_dir,'predictions.csv'))
  console.log(f"Accuracy:{accuracy}")
  console.save_text(os.path.join(output_dir,'logs.txt'))
  
  console.log("[Validation Completed.]\n")
  console.print(f"""[Model] Model saved @ {os.path.join(output_dir, "model_files")}\n""")
  console.print(f"""[Validation] Generation on Validation data saved @ {os.path.join(output_dir,'predictions.csv')}\n""")
  console.print(f"""[Logs] Logs saved @ {os.path.join(output_dir,'logs.txt')}\n""")

@click.command()
@click.option('--train-filename', default='../data/data0716/my_finetune_data/train.json')
@click.option('--learning-rate', '-lr', required=False, default=3e-5)
@click.option('--batch-size', '-bs', required=False, default=16)
@click.option('--model_name', required=False, default='alan-turing-institute/mt5-large-finetuned-mnli-xtreme-xnli')
@click.option('--output-dir', '-o', default='finetuned_models/alan_turing/')
@click.option('--epochs', '-e', required=False, default=6)
@click.option('--device-no', '-d', default='cuda:0')

def main(train_filename,
         learning_rate,
         batch_size,
         model_name,
         output_dir,
         epochs,
         device_no):

    model_2 = 'google/mt5-base'
    model_3 = 'google/mt5-small'
    #model parameters
    model_params={
    "MODEL":model_2,             # model_type: t5-base/t5-large
    "TRAIN_BATCH_SIZE":batch_size,          # training batch size
    "VALID_BATCH_SIZE":batch_size,          # validation batch size
    "TRAIN_EPOCHS":epochs,              # number of training epochs
    "VAL_EPOCHS":epochs,                # number of validation epochs
    "LEARNING_RATE":learning_rate,          # learning rate
    "MAX_SOURCE_TEXT_LENGTH":400,  # max length of source text
    "SEED": 42                     # set seed for reproducibility 
    }
    
    device = device_no

    #process json
    with open(train_filename,'r',encoding='utf-8') as filename:
        train = json.load(filename)
        filename.close()
    train_df = pd.DataFrame(train)
    train_df = process_dataframe(train_df)
    train_df['label'] = train_df['label'].astype(str)
    train_df, validation_df = train_test_split(train_df, shuffle=True, test_size = 0.05)
    
    
    MT5Trainer(train_dataframe=train_df, 
               dev_dataframe=validation_df, 
               source_text="stsb", 
               target_text = 'label',
               model_params=model_params, 
               device=device, 
               output_dir=output_dir)
    

if __name__ == '__main__':
    main()
