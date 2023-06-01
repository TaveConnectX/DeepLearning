import pandas as pd
import json


k = 3 # k번째 실험

# Read the json file
with open('loss_plot/experiment_{}/parameter sampling.json'.format(k)) as f:
    json_data = json.load(f)

# Extract the order, lr, batch_size, target_update, memory_len, and repeat_reward values to a list


data_list = []
for block in json_data:
    data = []
    for ele in block[0]:
        try:
            num = int(ele.split(':')[1])
        except:
            num = float(ele.split(':')[1])
        data.append(num)
    data.append(float(block[1]))

    data_list.append(data)

# Create a Pandas DataFrame from the list
df = pd.DataFrame(data_list, columns=['order', 'lr', 'batch_size', 'target_update', 'memory_len', 'repeat_reward', 'win_rate'])

# Write the DataFrame to an excel file
df.to_excel('loss_plot/experiment_{}/summary.xlsx'.format(k))