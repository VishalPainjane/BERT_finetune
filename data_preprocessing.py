from transformers import AutoTokenizer
import pandas as pd
import ast

def process_data(input_file, output_file):
    df = pd.read_csv(input_file)
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    questions = [q.strip() for q in df["question"]]
    context = [q.strip() for q in df["context"]]
    
    inputs = tokenizer(questions, context, max_length=384, truncation="only_second", padding="max_length", return_offsets_mapping=True)
    
    offset_mapping = inputs.pop("offset_mapping")
    
    start_positions = []
    end_positions = []
    
    answers = df["answers"]
    
    for i, offset in enumerate(offset_mapping):
        answer = ast.literal_eval(answers[i])
    
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer['text'][0])
        sequence_ids = inputs.sequence_ids(i)
    
        idx = 0
        while sequence_ids[idx] != 1:
            idx+=1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx+=1
        context_end = idx-1
        
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx)
            
            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx)
    
    data={'input_ids':inputs['input_ids'], 
          'attention_mask':inputs['attention_mask'], 
          'start_positions':start_positions, 
          'end_positions':end_positions}
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)

# Process both train and test data
process_data('data/squad_data_train.csv', 'data/encoding_train.csv')
process_data('data/squad_data_test.csv', 'data/encoding_test.csv')
