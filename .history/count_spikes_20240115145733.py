def _count_spikes(self, soma_voltage, threshold=0):
        spike_count = 0
        is_spiking = False

        for voltage in soma_voltage:
            if voltage > threshold and not is_spiking:
                is_spiking = True
                spike_count += 1
            elif voltage <= threshold:
                is_spiking = False

        return spike_count