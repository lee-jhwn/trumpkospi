train_cut = 0.8
w2v_size = 300
which_embedding = [None, 'Google_W2V', 'BERT']
which_embedding = which_embedding[2]
mode = ['lstm', 'bilstm', 'attention', 'BERT']
mode = mode[3]

print('===========config===========')
# print('train_cut')
print('embedding:', which_embedding)
print('mode:', mode)
print('============================')


if which_embedding == 'BERT':
    w2v_size = 768