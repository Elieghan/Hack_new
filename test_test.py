from tensorflow.keras.models import load_model


m = load_model('models/leonard_cohen')


f = open("models/u.txt", "r")
print(f.read())
