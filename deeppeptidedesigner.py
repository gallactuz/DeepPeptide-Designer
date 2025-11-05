import argparse
import numpy as np
from vae_model import VAEModel
from lstm_model import LSTMModel

class DeepPeptideDesigner:
    def __init__(self, vae_fasta, lstm_fastas, alpha=0.5, max_length=30):
        """Inicializa o DeepPeptideDesigner."""
        self.vae = VAEModel(vae_fasta, max_length=max_length)
        self.lstm_models = [LSTMModel(small_fasta=fasta, max_length=max_length) for fasta in lstm_fastas]
        self.alpha = alpha
        self.max_length = max_length
        self.aa_dict = None

    def initialize(self):
        """Carrega os dados e verifica compatibilidade."""
        X_vae, properties_vae = self.vae.load_data()
        self.aa_dict = self.vae.get_aa_dict()  # Usa o aa_dict fixo do VAE (20 aminoácidos)
        for lstm in self.lstm_models:
            lstm_data = lstm.load_data()
            # Força o aa_dict do LSTM a ser o mesmo do VAE
            lstm.aa_dict = self.aa_dict
            # Reconstroi os dados do LSTM com o novo aa_dict
            lstm_data = lstm.load_data_with_dict(self.aa_dict)
            lstm.train(lstm_data)
        self.vae.train(X_vae, properties_vae)

    def calculate_properties(self, sequence):
        """Calcula propriedades do peptídeo."""
        charge_dict = {'K': 1, 'R': 1, 'H': 0.1, 'D': -1, 'E': -1}
        mass_dict = {'A': 89.09, 'C': 121.16, 'D': 133.10, 'E': 147.13, 'F': 165.19,
                     'G': 75.07, 'H': 155.15, 'I': 131.17, 'K': 146.19, 'L': 131.17,
                     'M': 149.21, 'N': 132.12, 'P': 115.13, 'Q': 146.15, 'R': 174.20,
                     'S': 105.09, 'T': 119.12, 'V': 117.15, 'W': 204.23, 'Y': 181.19}
        hydrophobicity_dict = {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
                               'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
                               'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
                               'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3}
        charge = sum(charge_dict.get(aa, 0) for aa in sequence)
        mass = sum(mass_dict.get(aa, 110.0) for aa in sequence)
        hydrophobicity = sum(hydrophobicity_dict.get(aa, 0.0) for aa in sequence)
        def net_charge_at_pH(pH):
            pKa_dict = {'K': 10.5, 'R': 12.5, 'H': 6.0, 'D': 3.9, 'E': 4.3}
            ch = 0
            for aa in sequence:
                if aa in ['K', 'R', 'H']:
                    ch += 1 / (1 + 10**(pH - pKa_dict.get(aa, 7.0)))
                elif aa in ['D', 'E']:
                    ch -= 1 / (1 + 10**(pKa_dict.get(aa, 7.0) - pH))
            return ch
        pH_low, pH_high = 0.0, 14.0
        for _ in range(20):
            pH_mid = (pH_low + pH_high) / 2
            ch = net_charge_at_pH(pH_mid)
            if abs(ch) < 0.01:
                return np.array([charge, min(pH_mid, 14.0), mass, hydrophobicity])
            elif ch > 0:
                pH_low = pH_mid
            else:
                pH_high = pH_mid
        return np.array([charge, min((pH_low + pH_high) / 2, 14.0), mass, hydrophobicity])

    def combine_predictions(self, p_vae, p_lstms, length):
        """Combina as previsões do VAE e LSTMs."""
        p_lstm_avg = np.mean(p_lstms, axis=0) if len(p_lstms) > 1 else p_lstms[0]
        p_combined = self.alpha * p_vae + (1 - self.alpha) * p_lstm_avg
        sequence = []
        for probs in p_combined[:length]:
            aa_index = np.random.choice(len(self.aa_dict), p=probs)
            sequence.append(list(self.aa_dict.keys())[aa_index])
        return ''.join(sequence)

    def design_peptide(self, last_sequences=None, target_length=9, min_charge=1.0, min_hydrophobicity=0.0, num_peptides=5):
        """Gera uma lista de AMPs combinando VAE e LSTMs."""
        if not self.aa_dict:
            raise ValueError("Inicialize o designer com initialize() primeiro.")
        
        latent_sample = np.random.normal(size=(1, 100))  # Latent_dim fixo como 100 do VAE
        p_vae = self.vae.generate(latent_sample)
        if last_sequences is None or len(last_sequences) != len(self.lstm_models):
            last_sequences = [lstm.load_data()[-1] for lstm in self.lstm_models]
        p_lstms = [lstm.generate(seq) for lstm, seq in zip(self.lstm_models, last_sequences)]

        peptides = []
        max_attempts_per_peptide = 20
        for _ in range(num_peptides):  # Tenta gerar cada peptídeo
            attempts = 0
            while attempts < max_attempts_per_peptide:
                sequence = self.combine_predictions(p_vae, p_lstms, target_length)
                props = self.calculate_properties(sequence)
                if props[0] >= min_charge and props[3] >= min_hydrophobicity:
                    peptides.append((sequence, props))
                    break
                attempts += 1
            if attempts == max_attempts_per_peptide:
                print(f"Aviso: Não foi possível gerar o peptídeo {_ + 1} com os critérios desejados após {max_attempts_per_peptide} tentativas.")
                peptides.append((sequence, props))  # Adiciona a última tentativa mesmo que não atenda
                
        return peptides

def main():
    parser = argparse.ArgumentParser(description='Design AMPs using VAE and LSTM models.')
    parser.add_argument('-vae', type=str, required=True, help='FASTA file for the VAE big database.')
    parser.add_argument('-lstm', action='append', type=str, required=True, help='FASTA files for LSTM databases.')
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight for VAE vs LSTM predictions (0 to 1).')
    parser.add_argument('--length', type=int, default=9, help='Target length of the peptide.')
    parser.add_argument('--num', type=int, default=5, help='Number of peptides to generate (default: 5).')
    args = parser.parse_args()

    designer = DeepPeptideDesigner(vae_fasta=args.vae, lstm_fastas=args.lstm, alpha=args.alpha)
    designer.initialize()
    last_sequences = [lstm.load_data()[-1] for lstm in designer.lstm_models]
    peptides = designer.design_peptide(last_sequences=last_sequences, target_length=args.length, num_peptides=args.num)
    
    for i, (sequence, properties) in enumerate(peptides, 1):
        print(f"\nDesigned AMP #{i}: {sequence}")
        print(f"Properties - Charge: {properties[0]:.2f}, pI: {properties[1]:.2f}, "
              f"Mass: {properties[2]:.2f}, Hydrophobicity: {properties[3]:.2f}")

if __name__ == "__main__":
    main()
