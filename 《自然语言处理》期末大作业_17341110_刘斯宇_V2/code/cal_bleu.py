from nltk.translate.bleu_score import sentence_bleu

f=open('out.txt')
lines = f.readlines()
pre_list=[]
for line in lines:
    line = line.strip('\n')
    line_list = line.split(' ')
    pre_list.append(line_list)

f = open('test_target_1000.txt')
lines = f.readlines()
real_list=[]
for line in lines:
    line = line.strip('\n')
    line_list = line.split(' ')
    real_list.append(line_list)

for i in range(len(real_list)):
    score = sentence_bleu(real_list[i],pre_list[i])
    print(score)
