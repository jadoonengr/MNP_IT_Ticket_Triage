# !pip install transformers==3.2.0
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup, DistilBertModel, DistilBertTokenizer
import numpy as np
import torch
from datetime import datetime

def BERT_Tokenizer(model,tokenizer,text, max_len = 128):
    #Fix Token Lengths: Truncate or Pad to max BERT size (512)
    input_ids = []
    attention_masks = []
    
    #Tokenizing and Masking Text
    for row in text:
        encoded_dict = tokenizer.encode_plus(row,
                                             add_special_tokens = True, #add [CLS] and [SEP]
                                             padding = 'max_length',
                                             truncation = True,
                                             max_length = max_len, #pad and truncate
                                             return_attention_mask = True, #construct attention masks
                                            )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids_arr = np.asarray(input_ids)
    attention_masks_arr = np.asarray(input_ids)
    #Convert arrays to tensors
    input_ids_tensor = torch.tensor(input_ids_arr)  
    attention_mask_tensor = torch.tensor(attention_masks_arr)
    
    #print("Starting BERT Tokenization - may take a while depending on max_length_description (1 to 4 mins)")
    current_time = datetime.strptime(datetime.now().strftime("%M:%S"),"%M:%S")
    with torch.no_grad():
        last_hidden_states = model(input_ids_tensor, attention_mask_tensor)
    end_time = datetime.strptime(datetime.now().strftime("%M:%S"),"%M:%S")
    print("Total Time (mins):",str(end_time-current_time))

    text_embeddding = last_hidden_states[0][:,0,:].numpy()

    return text_embeddding