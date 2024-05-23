import re
import pandas as pd
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, f1_score
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D, Flatten
import matplotlib.pyplot as plt
from keras.models import model_from_json

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')


# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


# Preprocess text data
def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [token.lower() for token in tokens]
    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    preprocessed_text = " ".join(tokens)
    return preprocessed_text


# Create a CNN model for text classification
def create_text_cnn_model(input_shape):
    model = Sequential()

    model.add(Conv1D(128, 5, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Conv1D(128, 5, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(1, activation='sigmoid'))
    return model


# Train and visualize model
def train_model(model, x_train, y_train, x_test, y_test, batch_size=16, epochs=10):
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test))

    return history


def visualize_model_performance(history, x_test, y_test, model):
    y_pred_probabilities = model.predict(x_test)
    y_pred = (y_pred_probabilities > 0.5).astype(int)

    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    if history is not None:
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.show()

        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper right')
        plt.show()
    else:
        loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
        print(f"Test Loss: {loss}")
        print(f"Test Accuracy: {accuracy}")

# Save and load model
def save_model(model, model_name="text_cnn_model"):
    model_json = model.to_json()
    with open(f"{model_name}.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights(f"{model_name}.weights.h5")
    print(f"Model saved as {model_name}.json and {model_name}.h5")


def load_model(model_name="text_cnn_model"):
    try:
        with open(f"{model_name}.json", "r") as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(f"{model_name}.weights.h5")
        loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        print(f"Model loaded from {model_name}.json and {model_name}.h5")
        return loaded_model
    except FileNotFoundError:
        print("Model not found, training a new model...")
        return None


if __name__ == '__main__':
    # Load and preprocess data
    fake_data = load_data("data/fake.csv")
    real_data = load_data("data/true.csv")
    fake_data['category'] = 1
    real_data['category'] = 0
    data = pd.concat([fake_data, real_data], ignore_index=True)
    data['full_text'] = data['title'] + ' ' + data['text']
    data['full_text'] = data['full_text'].apply(preprocess_text)

    # Transform text data using TfidfVectorizer
    tokenizer = TfidfVectorizer(max_features=5000)
    X = tokenizer.fit_transform(data['full_text']).toarray()
    y = data['category'].values

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Reshape data for CNN input
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Create and train model
    input_length = (X_train.shape[1], 1)

    model = load_model()
    if model is None:
        model = create_text_cnn_model(input_length)
        history = train_model(model, X_train, y_train, X_test, y_test)
        save_model(model)
        visualize_model_performance(history, X_test, y_test, model)
    else:
        visualize_model_performance(None, X_test, y_test, model)

    print(model.summary())
