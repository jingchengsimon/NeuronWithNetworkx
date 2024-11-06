# Choose the time point with the maximum soma voltage
        # Find the index that soma_v first cross 0
        max_soma_idx = np.argmax(soma_v) - 4
        print(f"Max soma idx: {max_soma_idx}")

        v_vals = [seg_v[i][max_soma_idx] for i in range(len(seg_v))]

        # Reset the voltage
        seg_list = [seg for sec in h.allsec() for seg in sec]
        for i in range(len(seg_list)):
            seg_list[i].v = v_vals[i]

        ps = h.PlotShape(False) # default variable is voltage
        ps.variable('v')
        # ps.scale(min(v_vals), max(v_vals))
        ps.scale(-70, 0)
        
        # Create a custom colormap using Matplotlib (cool colormap)
        cmap = cm.cool
        seg_syn_df = self.section_synapse_df[self.section_synapse_df['type'] == 'C']['segment_synapse'].values
        soma = self.section_synapse_df[self.section_synapse_df['region'] == 'soma']['section_synapse'].values[0]
        fig1=ps.plot(plotly, cmap=cmap).mark(soma(0.5))

        fig2=ps.plot(plotly, cmap=cmap).mark(soma(0.5))
        for seg_syn in seg_syn_df:
            fig2 = fig2.mark(seg_syn)
        
        # Set the axis limits for x, y, and z to be the same
        axis_limit = [-200, 1200] 
        fig1.update_layout(scene=dict(
            xaxis=dict(range=axis_limit),
            yaxis=dict(range=axis_limit),
            zaxis=dict(range=axis_limit)
        ))

        fig2.update_layout(scene=dict(
            xaxis=dict(range=axis_limit),
            yaxis=dict(range=axis_limit),
            zaxis=dict(range=axis_limit)
        ))

        # Create a colormap function
        colormap = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin=0, vmax=1)).to_rgba
        plotly_colorscale = [[v, f'rgb{tuple(int(255 * c) for c in colormap(v)[:3])}'] for v in np.linspace(0, 1, cmap.N)]
        colorbar_trace = go.Scatter(
            x=[0],
            y=[0],
            mode='markers',
            marker=dict(
                colorscale=plotly_colorscale,
                cmin=-70, #min(v_vals),
                cmax=0, #max(v_vals),
                colorbar=dict(
                    title='v (mV)',
                    thickness=20  # Adjust the thickness of the colorbar
                ),
                showscale=True
            )
        )

        # Add the colorbar trace to the figure
        fig1.add_trace(colorbar_trace)
        fig1.update_xaxes(showticklabels=False, showgrid=False)
        fig1.update_yaxes(showticklabels=False, showgrid=False)
        fig1.update_layout(plot_bgcolor='rgba(0,0,0,0)')

        fig2.add_trace(colorbar_trace)  
        fig2.update_xaxes(showticklabels=False, showgrid=False)
        fig2.update_yaxes(showticklabels=False, showgrid=False)
        fig2.update_layout(plot_bgcolor='rgba(0,0,0,0)')

        fig1.show()
        fig2.show()