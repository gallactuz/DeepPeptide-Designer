import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Flatten, Reshape, Conv1D, MaxPooling1D, UpSampling1D, Concatenate, Layer
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from Bio import SeqIO

class VAEModel:
    def __init__(self, large_fasta, max_length=None, latent_dim=100):
        """Inicializa o modelo VAE."""
        self.large_fasta = large_fasta
        self.max_length = max_length if max_length else self._determine_max_length()
        self.latent_dim = latent_dim
        self.aa_dict = {'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
                        'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19}
        self.amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
        self.model = None
        self.encoder = None
        self.decoder = None
        self.min_max_values = None  # Armazenará os valores min/max calculados

    def _determine_max_length(self):
        """Determina o comprimento máximo das sequências no FASTA."""
        sequences = [str(record.seq) for record in SeqIO.parse(self.large_fasta, "fasta")]
        return max(len(seq) for seq in sequences)

    def load_data(self):
        """Carrega e codifica o banco grande de AMPs, calculando min/max dinamicamente."""
        sequences = [str(record.seq) for record in SeqIO.parse(self.large_fasta, "fasta")]
        X = np.array([self._one_hot_encode(seq) for seq in sequences])

        # Calcula as propriedades brutas para todas as sequências
        charges = [sum({'K': 1, 'R': 1, 'H': 0.1, 'D': -1, 'E': -1}.get(aa, 0) for aa in seq) for seq in sequences]
        pIs = [sum({'K': 10.5, 'R': 12.5, 'H': 6.0, 'D': 3.9, 'E': 4.3}.get(aa, 7.0) for aa in seq) for seq in sequences]
        masses = [sum({'A': 89.09, 'C': 121.16, 'D': 133.10, 'E': 147.13, 'F': 165.19, 'G': 75.07, 'H': 155.15,
                       'I': 131.17, 'K': 146.19, 'L': 131.17, 'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15,
                       'R': 174.20, 'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19}.get(aa, 110.0) for aa in seq) for seq in sequences]
        hydrophobicities = [sum({'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2,
                                 'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5,
                                 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3}.get(aa, 0.0) for aa in seq) for seq in sequences]

        # Determina os valores min/max dinamicamente
        self.min_max_values = {
            'charge': (min(charges), max(charges)),
            'pI': (min(pIs), max(pIs)),
            'mass': (min(masses), max(masses)),
            'hydrophobicity': (min(hydrophobicities), max(hydrophobicities))
        }

        # Garante que min e max sejam diferentes para evitar divisão por zero
        for prop in self.min_max_values:
            if self.min_max_values[prop][0] == self.min_max_values[prop][1]:
                self.min_max_values[prop] = (self.min_max_values[prop][0] - 1, self.min_max_values[prop][1] + 1)

        # Normaliza as propriedades usando os valores calculados
        properties = np.array([self._calculate_properties(seq) for seq in sequences])
        return X, properties

    def _one_hot_encode(self, sequence):
        """Codifica uma sequência em one-hot."""
        encoding = np.zeros((self.max_length, len(self.aa_dict)))
        for i, aa in enumerate(sequence[:self.max_length]):
            if aa in self.aa_dict:
                encoding[i, self.aa_dict[aa]] = 1
            else:
                encoding[i, :] = 0.25  # Valor padrão
        return encoding

    def _normalize(self, value, min_value, max_value):
        """Normaliza um valor entre min e max."""
        return (value - min_value) / (max_value - min_value)

    def _calculate_properties(self, sequence):
        """Calcula propriedades normalizadas usando min/max do banco de dados."""
        charge = sum({'K': 1, 'R': 1, 'H': 0.1, 'D': -1, 'E': -1}.get(aa, 0) for aa in sequence)
        pI = sum({'K': 10.5, 'R': 12.5, 'H': 6.0, 'D': 3.9, 'E': 4.3}.get(aa, 7.0) for aa in sequence)
        mass = sum({'A': 89.09, 'C': 121.16, 'D': 133.10, 'E': 147.13, 'F': 165.19, 'G': 75.07, 'H': 155.15,
                    'I': 131.17, 'K': 146.19, 'L': 131.17, 'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15,
                    'R': 174.20, 'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19}.get(aa, 110.0) for aa in sequence)
        hydrophobicity = sum({'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2,
                              'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5,
                              'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3}.get(aa, 0.0) for aa in sequence)

        # Normalização usando os valores min/max calculados em load_data
        charge_norm = self._normalize(charge, *self.min_max_values['charge'])
        pI_norm = self._normalize(pI, *self.min_max_values['pI'])
        mass_norm = self._normalize(mass, *self.min_max_values['mass'])
        hydrophobicity_norm = self._normalize(hydrophobicity, *self.min_max_values['hydrophobicity'])

        return np.array([charge_norm, pI_norm, mass_norm, hydrophobicity_norm])

    def build_model(self):
        """Constrói o modelo VAE."""
        input_seq = Input(shape=(self.max_length, len(self.aa_dict)))
        input_properties = Input(shape=(4,))
        x = Conv1D(32, 3, activation='relu', padding='same')(input_seq)
        x = MaxPooling1D(2, padding='same')(x)
        x = Conv1D(64, 3, activation='relu', padding='same')(x)
        x = MaxPooling1D(2, padding='same')(x)
        x = Flatten()(x)
        x = Concatenate()([x, input_properties])
        h = Dense(256, activation='relu')(x)
        z_mean = Dense(self.latent_dim)(h)
        z_log_var = Dense(self.latent_dim)(h)

        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=(K.shape(z_mean)[0], self.latent_dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon

        z = Lambda(sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])
        self.encoder = Model([input_seq, input_properties], z)

        decoder_input = Input(shape=(self.latent_dim,))
        h_decoded = Dense(256, activation='relu')(decoder_input)
        h_decoded = Dense((self.max_length // 4 + (self.max_length % 4 > 0)) * 64, activation='relu')(h_decoded)
        h_decoded = Reshape((self.max_length // 4 + (self.max_length % 4 > 0), 64))(h_decoded)
        x = UpSampling1D(2)(h_decoded)
        x = Conv1D(64, 3, activation='relu', padding='same')(x)
        x = UpSampling1D(2)(x)
        decoded = Conv1D(len(self.aa_dict), 3, activation='softmax', padding='same')(x)
        decoded = decoded[:, :self.max_length, :]
        self.decoder = Model(decoder_input, decoded)

        class VAELossLayer(Layer):
            def __init__(self, **kwargs):
                super(VAELossLayer, self).__init__(**kwargs)

            def call(self, inputs):
                input_seq, input_properties, vae_output, z_mean, z_log_var = inputs
                reconstruction_loss = K.sum(K.square(input_seq - vae_output), axis=(1, 2))
                kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
                vae_loss = K.mean(reconstruction_loss + kl_loss)
                self.add_loss(vae_loss)
                return vae_output

        vae_output = self.decoder(z)
        vae_loss_layer = VAELossLayer()([input_seq, input_properties, vae_output, z_mean, z_log_var])
        self.model = Model([input_seq, input_properties], vae_loss_layer)
        self.model.compile(optimizer=Adam())

    def train(self, X, properties, epochs=50, batch_size=32):
        """Treina o VAE."""
        if not self.model:
            self.build_model()
        self.model.fit([X, properties], epochs=epochs, batch_size=batch_size, verbose=1)

    def generate(self, latent_sample=None):
        """Gera probabilidades de aminoácidos a partir do espaço latente."""
        if not self.decoder:
            self.build_model()
        if latent_sample is None:
            latent_sample = np.random.normal(size=(1, self.latent_dim))
        pred = self.decoder.predict(latent_sample, verbose=0)[0]
        return pred

    def get_aa_dict(self):
        """Retorna o dicionário de aminoácidos."""
        return self.aa_dict
