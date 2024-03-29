import os
import matplotlib.pyplot as plt
def visualize_synapses(self, folder_path, title='Synapses'):
        s = h.PlotShape(False)
        self._recursive_plot(s, self.section_synapse_df['segment_synapse'].values)
        plt.title(title)
        
        file_path = os.path.join(folder_path, f'figure_synapses.png')
        plt.savefig(file_path)
        plt.close()