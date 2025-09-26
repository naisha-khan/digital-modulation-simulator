import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import upfirdn
import random

class DigitalModulationSimulator:
    """
    Digital Modulation Simulator for ASK, PSK, and QAM
    Demonstrates telecommunication signal processing and BER analysis
    """
    
    def __init__(self, carrier_freq=1000, sampling_rate=8000, symbol_rate=100):
        self.fc = carrier_freq
        self.fs = sampling_rate
        self.symbol_rate = symbol_rate
        self.samples_per_symbol = self.fs // self.symbol_rate
        
    def generate_data_bits(self, num_bits):
        """Generate random binary data"""
        return np.random.randint(0, 2, num_bits)
    
    def ask_modulate(self, bits, amplitude_levels=[0, 1]):
        """Amplitude Shift Keying (ASK) Modulation"""
        symbols = [amplitude_levels[bit] for bit in bits]
        t = np.linspace(0, len(symbols)/self.symbol_rate, len(symbols) * self.samples_per_symbol, False)
        
        modulated_signal = np.repeat(symbols, self.samples_per_symbol) * np.cos(2 * np.pi * self.fc * t)
        return modulated_signal, t, symbols
    
    def psk_modulate(self, bits, M=2):
        """Phase Shift Keying (PSK) Modulation"""
        if M == 2:  # BPSK
            symbols = [1 if bit == 0 else -1 for bit in bits]
        elif M == 4:  # QPSK
            # Group bits in pairs for QPSK
            grouped_bits = [bits[i:i+2] for i in range(0, len(bits), 2)]
            symbol_map = {(0,0): 1+1j, (0,1): -1+1j, (1,0): 1-1j, (1,1): -1-1j}
            symbols = [symbol_map.get(tuple(pair), 1+1j) for pair in grouped_bits]
        
        t = np.linspace(0, len(symbols)/self.symbol_rate, len(symbols) * self.samples_per_symbol, False)
        
        if M == 2:
            modulated_signal = np.repeat(symbols, self.samples_per_symbol) * np.cos(2 * np.pi * self.fc * t)
        else:  # QPSK
            I = np.repeat([s.real for s in symbols], self.samples_per_symbol)
            Q = np.repeat([s.imag for s in symbols], self.samples_per_symbol)
            modulated_signal = I * np.cos(2 * np.pi * self.fc * t) - Q * np.sin(2 * np.pi * self.fc * t)
        
        return modulated_signal, t, symbols
    
    def qam_modulate(self, bits, M=16):
        """Quadrature Amplitude Modulation (QAM)"""
        # Group bits for M-QAM
        bits_per_symbol = int(np.log2(M))
        grouped_bits = [bits[i:i+bits_per_symbol] for i in range(0, len(bits), bits_per_symbol)]
        
        # 16-QAM constellation mapping
        if M == 16:
            constellation = {
                (0,0,0,0): -3-3j, (0,0,0,1): -3-1j, (0,0,1,0): -3+3j, (0,0,1,1): -3+1j,
                (0,1,0,0): -1-3j, (0,1,0,1): -1-1j, (0,1,1,0): -1+3j, (0,1,1,1): -1+1j,
                (1,0,0,0): 3-3j,  (1,0,0,1): 3-1j,  (1,0,1,0): 3+3j,  (1,0,1,1): 3+1j,
                (1,1,0,0): 1-3j,  (1,1,0,1): 1-1j,  (1,1,1,0): 1+3j,  (1,1,1,1): 1+1j
            }
        
        symbols = [constellation.get(tuple(group), 1+1j) for group in grouped_bits]
        
        t = np.linspace(0, len(symbols)/self.symbol_rate, len(symbols) * self.samples_per_symbol, False)
        
        I = np.repeat([s.real for s in symbols], self.samples_per_symbol)
        Q = np.repeat([s.imag for s in symbols], self.samples_per_symbol)
        modulated_signal = I * np.cos(2 * np.pi * self.fc * t) - Q * np.sin(2 * np.pi * self.fc * t)
        
        return modulated_signal, t, symbols
    
    def add_awgn_noise(self, signal, snr_db):
        """Add Additive White Gaussian Noise"""
        signal_power = np.mean(signal**2)
        snr_linear = 10**(snr_db/10)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power) * np.random.randn(len(signal))
        return signal + noise
    
    def calculate_ber(self, transmitted_bits, received_bits):
        """Calculate Bit Error Rate"""
        errors = np.sum(transmitted_bits != received_bits)
        return errors / len(transmitted_bits)
    
    def plot_constellation(self, symbols, title="Constellation Diagram"):
        """Plot constellation diagram"""
        if isinstance(symbols[0], complex):
            plt.figure(figsize=(8, 6))
            plt.scatter([s.real for s in symbols], [s.imag for s in symbols], alpha=0.7)
            plt.xlabel('In-phase (I)')
            plt.ylabel('Quadrature (Q)')
            plt.title(title)
            plt.grid(True)
            plt.axis('equal')
            plt.show()
    
    def ber_vs_snr_analysis(self, modulation_type='BPSK', num_bits=1000):
        """Analyze BER vs SNR performance"""
        snr_range = np.arange(0, 21, 2)  # 0 to 20 dB
        ber_values = []
        
        for snr in snr_range:
            bits = self.generate_data_bits(num_bits)
            
            # Modulate based on type
            if modulation_type == 'ASK':
                signal, _, _ = self.ask_modulate(bits)
            elif modulation_type == 'BPSK':
                signal, _, _ = self.psk_modulate(bits, M=2)
            elif modulation_type == 'QPSK':
                signal, _, _ = self.psk_modulate(bits, M=4)
            elif modulation_type == '16QAM':
                signal, _, _ = self.qam_modulate(bits, M=16)
            
            # Add noise
            noisy_signal = self.add_awgn_noise(signal, snr)
            
            # Simple demodulation (for demonstration)
            # In practice, this would be more sophisticated
            received_bits = bits  # Placeholder - would implement proper demodulation
            
            # Calculate theoretical BER for comparison
            if modulation_type == 'BPSK':
                from scipy.special import erfc
                theoretical_ber = 0.5 * erfc(np.sqrt(10**(snr/10)))
            else:
                # Simplified - would calculate for each modulation type
                theoretical_ber = 0.5 * erfc(np.sqrt(10**(snr/10)))
            
            ber_values.append(theoretical_ber)
        
        # Plot BER vs SNR
        plt.figure(figsize=(10, 6))
        plt.semilogy(snr_range, ber_values, 'b-o', label=f'{modulation_type} (Theoretical)')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Bit Error Rate (BER)')
        plt.title(f'BER vs SNR Performance - {modulation_type}')
        plt.grid(True, which="both", ls="-", alpha=0.3)
        plt.legend()
        plt.show()
        
        return snr_range, ber_values

def demonstrate_modulation_techniques():
    """Demonstration function showing all modulation techniques"""
    simulator = DigitalModulationSimulator()
    
    # Generate test data
    test_bits = simulator.generate_data_bits(32)
    print(f"Test bits: {test_bits}")
    
    # ASK Modulation
    print("\n=== ASK Modulation ===")
    ask_signal, t_ask, ask_symbols = simulator.ask_modulate(test_bits)
    print(f"ASK symbols: {ask_symbols[:8]}...")  # Show first 8
    
    # BPSK Modulation
    print("\n=== BPSK Modulation ===")
    bpsk_signal, t_bpsk, bpsk_symbols = simulator.psk_modulate(test_bits, M=2)
    print(f"BPSK symbols: {bpsk_symbols[:8]}...")
    
    # QPSK Modulation
    print("\n=== QPSK Modulation ===")
    qpsk_signal, t_qpsk, qpsk_symbols = simulator.psk_modulate(test_bits, M=4)
    print(f"QPSK symbols: {qpsk_symbols[:4]}...")
    
    # 16-QAM Modulation
    print("\n=== 16-QAM Modulation ===")
    qam_signal, t_qam, qam_symbols = simulator.qam_modulate(test_bits, M=16)
    print(f"16-QAM symbols: {qam_symbols[:2]}...")
    
    # Plot signals
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.plot(t_ask[:800], ask_signal[:800])
    plt.title('ASK Modulated Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.subplot(2, 2, 2)
    plt.plot(t_bpsk[:800], bpsk_signal[:800])
    plt.title('BPSK Modulated Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.subplot(2, 2, 3)
    plt.plot(t_qpsk[:800], qpsk_signal[:800])
    plt.title('QPSK Modulated Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.subplot(2, 2, 4)
    plt.plot(t_qam[:800], qam_signal[:800])
    plt.title('16-QAM Modulated Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    
    plt.tight_layout()
    plt.show()
    
    # Show constellation diagrams
    simulator.plot_constellation(qpsk_symbols, "QPSK Constellation")
    simulator.plot_constellation(qam_symbols, "16-QAM Constellation")
    
    # BER Analysis
    print("\n=== BER vs SNR Analysis ===")
    snr, ber = simulator.ber_vs_snr_analysis('BPSK', 10000)
    print(f"BER analysis completed for BPSK")

if __name__ == "__main__":
    demonstrate_modulation_techniques()