import matplotlib.pylab as plt
import pickle

def draw_graph(filename, title, max_epoch=None):



    with open(filename, 'rb') as f:
        history = pickle.load(f)

    if max_epoch is not None:
        for k in history.keys():
            history[k] = history[k][:max_epoch]

    fig, loss_ax = plt.subplots()
    corr_ax = loss_ax.twinx()
    corr_ax.set_ylim(-0.2, 1.0)
    loss_ax.set_ylim(0, 1.2)

    loss_ax.plot(history['loss'], 'y', label='train loss')
    loss_ax.plot(history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')

    corr_ax.plot(history['train_corr'], 'b', label='train corr')
    corr_ax.plot(history['test_corr'], 'g', label='test corr')
    corr_ax.set_ylabel('correlation')
    corr_ax.legend(loc='upper right')

    plt.title(title)


    plt.show()
    return






draw_graph('bilstm_history.pkl', 'Bi-LSTM')
draw_graph('attention_history.pkl', 'Attention')
draw_graph('bert_feature.pkl', 'BERT Feature-ext')
draw_graph('bert_finetuning.pkl', 'BERT Fine-tuning', max_epoch=50)