#importing necessary libraries
import pandas as pd
import os


#create lists
names=[]
labels=[]

#open file
with open("trial_metadata.txt") as fp:
    for line in fp:

        line=line.strip()

        print(line)
        splitted=line.split()
        #perform indexing
        name=splitted[1]
        label=splitted[5]

        print(name)
        print(label)
        #perform appending
        names.append(name)
        labels.append(label)


#create dictionary
my_dict={'Name':names,'Label':labels}
#convert it into dataframe
df = pd.DataFrame(my_dict)

# Display DataFrame
print("Created DataFrame:\n",df,"\n")

#save dataframe
df.to_csv("Project_Dataset/data_info.csv",index=False)

#read dataframe
final_df=pd.read_csv("Project_Dataset/data_info.csv")

human_voice=final_df[final_df['Label']=='bonafide']
#save dataframe
human_voice.to_csv("Project_Dataset/human_voice.csv",index=False)

deepfake_voice=final_df[final_df['Label']=='spoof']
#save dataframe
deepfake_voice.to_csv("Project_Dataset/deepfake_voice.csv",index=False)
