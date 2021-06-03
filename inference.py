from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
from flask import Flask
from flask import request
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer

Nl = 30

app = Flask(__name__)

@app.route('/poemlt')
def gen_poemlt():
    k1 = str(request.args.get('singer'))
    # k2 = str(request.args.get('first_words'))
    # feat = np.array([k1,k2]).reshape(1, -1)
    try:
        filename = 'models/' + k1
        model = load_model(filename)
    except:
        return "Sorry, this singer is not available yet."

    txtfile = 'models/' + k1 + '/' + k1 + '.txt'
    f = open(txtfile, "r")
    text_em = f.read()
    list_text = text_em.split('\n')
    df = pd.DataFrame(columns=['lyrics'])
    for line in list_text:
        df = df.append({'lyrics': line}, ignore_index=True)
    df['cleaned'] = df['cleaned'].map(lambda x: '^' + x + '$')
    list_cl_q = ' '.join(df['cleaned'].tolist())
    list_cl_char = list(set(list(list_cl_q)))
    in_dim = len(list_cl_char)
    sample_tokenizer = Tokenizer(char_level=True, lower=False)
    sample_tokenizer.fit_on_texts(df.cleaned)
    list_lyrics = []
    for _ in range(Nl):
        list_lyrics.append(generate_text(0.4, model, sample_tokenizer, in_dim))
    return list_lyrics


def generate_text(temperature, model,sample_tokenizer,in_dim):
    char_pred = '^'
    sequence = np.zeros(99)
    text = [char_pred]
    sequence[0] = sample_tokenizer.word_index[char_pred]
    lenght = len(text)
    while char_pred != '$' and lenght < 99:
        y_pred_pad = model.predict(sequence.reshape(1, -1))
        y_pred = y_pred_pad[0][lenght - 1][1:] / y_pred_pad[0][lenght - 1][1:].sum()
        y_pred = y_pred ** (1 / temperature)
        y_pred = y_pred / y_pred.sum()
        car = np.random.choice(in_dim - 1, p=y_pred.ravel()) + 1
        if lenght < 99: sequence[lenght] = car
        char_pred = sample_tokenizer.index_word[car]
        text.append(char_pred)
        lenght = len(text)
    del text[-1], text[0]
    return ''.join(text)


def main():
    """
    Run the test function and predict_churn
    Author:  Elie Ghanassia
    """
    port = os.environ.get('PORT')
    print(port)
    app.run(host='0.0.0.0', port=int(port))


if __name__ == '__main__':
    main()
