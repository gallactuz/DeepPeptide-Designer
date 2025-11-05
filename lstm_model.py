import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Masking, Dropout
import tensorflow as tf
from Bio import SeqIO

tf.config.run_functions_eagerly(True)

class LSTMModel:
    def __init__(self, small_fasta, max_length=30):
        """Inicializa o modelo LSTM."""
        self.small_fasta = small_fasta
        self.max_length = max_length
        self.aa_dict = None
        self.model = None

    def load_data(self):
        """Carrega e codifica o banco pequeno de AMPs com um aa_dict dinâmico."""
        sequences = list(SeqIO.parse(self.small_fasta, 'fasta'))
        if len(sequences) < 2:
            raise ValueError("O arquivo FASTA deve conter pelo menos 2 sequências para treinamento.")
        if self.aa_dict is None:
            amino_acids = sorted(set(''.join(str(record.seq) for record in sequences)))
            self.aa_dict = {aa: i for i, aa in enumerate(amino_acids)}
        encoded_sequences = [self._one_hot_encode(str(record.seq)) for record in sequences]
        return np.array(encoded_sequences)

    def load_data_with_dict(self, aa_dict):
        """Carrega e codifica o banco pequeno de AMPs usando um aa_dict específico."""
        self.aa_dict = aa_dict
        sequences = list(SeqIO.parse(self.small_fasta, 'fasta'))
        if len(sequences) < 2:
            raise ValueError("O arquivo FASTA deve conter pelo menos 2 sequências para treinamento.")
        encoded_sequences = [self._one_hot_encode(str(record.seq)) for record in sequences]
        return np.array(encoded_sequences)

    def _one_hot_encode(self, sequence):
        """Codifica uma sequência em one-hot usando o aa_dict atual."""
        encoding = np.zeros((self.max_length, len(self.aa_dict)))
        for i, aa in enumerate(sequence[:self.max_length]):
            if aa in self.aa_dict:
                encoding[i, self.aa_dict[aa]] = 1
        return encoding

    def build_model(self, num_amino_acids):
        """Constrói o modelo LSTM."""
        self.model = Sequential()
        self.model.add(Masking(mask_value=0, input_shape=(self.max_length, num_amino_acids)))
        self.model.add(LSTM(128, return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(num_amino_acids, activation='softmax'))
        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    def train(self, sequences, epochs=200, batch_size=32):
        """Treina o modelo LSTM."""
        if len(sequences) < 2:
            raise ValueError("Número insuficiente de sequências para treinamento.")
        if not self.model:
            self.build_model(len(self.aa_dict))
        X, y = sequences[:-1], sequences[1:]
        history = self.model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=1)
        return history

    def generate(self, last_sequence):
        """Gera probabilidades de aminoácidos a partir da última sequência."""
        if not self.model:
            raise ValueError("Modelo LSTM não construído. Execute train() primeiro.")
        pred = self.model.predict(last_sequence.reshape(1, self.max_length, len(self.aa_dict)), verbose=0)[0]
        return pred

    def get_aa_dict(self):
        """Retorna o dicionário de aminoácidos."""
        return self.aa_dict
