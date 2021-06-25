import csv
import logging
import numpy as np
from collections import deque

class NeuralNetMonitor:
    """ Monitor of the neural network variables.
    ==============================================================
    - Params:
        ensembles [dict] : dict mapping ensemble names to number of neurons.
        stimuli [dict] : dict mapping stimuli names to number of stimuli components.
        encoded_inputs [dict] : dict encoded stimuli names to number of input neurons.
        output_neuron_names [list] : list of motor ensembles.
    ==============================================================
    """
    def __init__(self, ensembles, stimuli, encoded_inputs, output_neuron_names):
        self.ensembles = ensembles
        self.stimuli = stimuli
        self.encoded_inputs = encoded_inputs
        self.output_neurons = output_neuron_names
        self.neuron_names = [name + '_' + str(i) for name, val in ensembles.items() for i in range(val)]
        self.stimuli_names = [name + '_' + str(i) for name, val in stimuli.items() for i in range(val)]
        self.encoded_input_names = [name + '_' + str(i) for name, val in encoded_inputs.items() for i in range(val)]
        self.output_names = [name + '_' + str(i) for name, val in ensembles.items()\
                            for i in range(val) if name in self.output_neurons]
        
        self.records = {key1 : {key2 : deque([]) for key2 in self.neuron_names}\
            for key1 in ['spikes', 'voltages', 'currents', 'outputs', 'recovery', 'neuron_theta']}
        self.records.update({'stimuli' : {key : deque([]) for key in self.stimuli_names}})
        self.records.update({'encoded_inputs' : {key : deque([]) for key in self.encoded_input_names}})
        # Currently only activities of motor neurons implemented
        self.records.update({'activities' : {key : deque([]) for key in self.output_names}})

    def update(self, **params):
        """ Updates the recorded data with the new sample.
        """
        for key, value in params.items():
            if key in self.records.keys():
                for node_name, node_val in zip(self.records[key].keys(), value):
                    self.records[key][node_name].append(node_val)
            else:
                logging.warning('Variable {} is not in available '\
                    'variables to be recorded'.format(key))

    def get(self, varname, node_name=None):
        """ Get the recorded values of a queried variable. It also allows the 
        retrieval of a particular neuron or input node. If no neuron is fixed, it 
        returns the records of all the records of the requested var.
        ===========================================================================
        - Args:
            varname [str] : name of the monitored variable.
            node_name [str or None] : name of the node or neuron.
        - Returns:
        ===========================================================================
        """
        if varname not in self.records.keys():
            raise Exception(logging.error('Variable {} does not exist or is not recorded. '\
                'Recorded variables are {}.'.format(varname, tuple(self.records.keys()))))
        if node_name is not None and node_name not in self.records[varname].keys():
            raise Exception(logging.error('Node/Neuron {} does not exist for variable {}. '\
                'Existing nodes are {}'.format(node_name, varname, tuple(self.records[varname].keys()) )))
        var_records = self.records[varname].copy()
        if node_name is not None:
            return np.hstack(var_records[node_name])
        else:
            return {key: np.hstack(val) for key, val in var_records.items()}
    
    def get_ensemble(self, varname, ensemble_name):
        #TODO
        # if varname not in self.records.keys():
        #     raise Exception(logging.error('Variable {} does not exist or is not recorded. '\
        #         'Recorded variables are {}'.format(varname, tuple(self.records.keys()))))
        # if ensemble_name not in self.records[varname].keys():
        #     raise Exception(logging.error('Ensemble {} does not exist {}. '\
        #         'Existing ensembles are {}'.format(ensemble_name, tuple(self.records[varname].keys())))
        import pdb; pdb.set_trace()
        return {'_'.join(key.split('_')[:-1]) : val for key, val in self.records[varname].items()}[ensemble_name]

    def reset(self):
        """ Clear all the monitors. """
        for varname, var_data in self.records.items():
            for node_name, node in var_data.items():
                self.records[varname][node_name].clear()

    def __len__(self):
        return len(tuple(self.records['voltages'].values())[0])

    # def to_csv(self, csv_name, csv_path):
    #     # cols = [ for var_names in iter([self.get_inputs()])]
    #     cols = []
    #     for var_name, var in self.data.items():
    #         for k in var[0]:
    #             cols.extend([var_name+'_'+k[0]+'_'+str(k[1])])
    #     data = np.hstack([var[1] for var in self.data.values()])
    #     pd.DataFrame(data, columns=cols).to_csv(csv_path + csv_name+'.csv') 
    #     import pdb; pdb.set_trace()

