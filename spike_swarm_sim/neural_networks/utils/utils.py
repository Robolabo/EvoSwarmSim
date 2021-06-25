import logging
import inspect
from spike_swarm_sim.utils import merge_dicts, any_duplicates, ConfigException
import spike_swarm_sim.register as reg
from spike_swarm_sim.neural_networks.receptive_field import ReceptiveField
from spike_swarm_sim.neural_networks.decoding import Decoder
from spike_swarm_sim.neural_networks.neuron_models import SpikingNeuronModel

#* ---- CONFIGURATION CHECK FUNCTIONS ---- #
def ensembles_checker(topology):
    ensembles_config = topology['ensembles']
    neuron_model = topology['neuron_model']
    if len(ensembles_config) == 0:
        raise ConfigException('No ANN ensemble specified.')
    for name, ens in ensembles_config.items():
        if 'n' in ens and ens['n'] <= 0.:
            raise ConfigException('Ensemble neurons must be greater than 0. '\
                'Error in ensemble {}'.format(name))
        neurons_inspect = inspect.getfullargspec(reg.neuron_models[neuron_model].add)
        neuron_args = {key : val for key, val in zip(neurons_inspect.args[1:], neurons_inspect.defaults)}
        if 'params' in ens:
            for p_name, param in ens['params'].items():
                if p_name not in neuron_args:
                    raise ConfigException('The parameter "{}" does not exist for neuron model {} '\
                        'of ensemble {}. Available neuron model arguments are {}.'\
                        .format(p_name, neuron_model, name, tuple(neuron_args.keys())))
                if type(param) != type(neuron_args[p_name]):
                    raise ConfigException(' Wrong type of parameter {} in ensemble {}. Type {} '\
                        'should have been received.'.format(p_name, name, type(neuron_args[p_name]).__name__))

def synapses_checker(topology):
    synapse_config = topology['synapses']
    ensembles = topology['ensembles']
    synapse_params = topology['synapse_params'] if 'synapse_params' in topology else None
    if len(synapse_config) == 0:
        raise ConfigException('No ANN synapses specified.')
    for name, syn in synapse_config.items():
        # Check types
        for var, expected_type in zip(['pre', 'post', 'p', 'ntx_fixed', 'neuroTX', 'trainable'],\
                [str, str, float, bool, str, bool]):
            if var in syn and not isinstance(syn[var], expected_type):
                raise ConfigException('Synapse field "{}" must be of type {} and its type is {}.'\
                    .format(var, expected_type.__name__, type(syn[var]).__name__))
        if syn['pre'] not in merge_dicts([ensembles, topology['stimuli']]):
            raise ConfigException('Presynaptic ensemble "{}" of synapse "{}" does not '\
                'exist in the defined ensembles.'.format(syn['pre'], name))
        if syn['post'] not in ensembles:
            raise ConfigException('Postsynaptic ensemble "{}" of synapse "{}" does not '\
                'exist in the defined ensembles.'.format(syn['post'], name))
        if 'p' in syn and not 0 <= syn['p'] <= 1:
            raise ConfigException('Connection probability of synapse {} must be in the interval [0, 1]'.format(name))
        if 'neuroTX' in syn and syn['neuroTX'] not in ['AMPA', 'GABA', 'NDMA', 'AMPA+NDMA']:
            raise ConfigException('Neurotransmitter {} of synapse {} is not implemented. '\
                'Available neurotransmitters are {}.'.format(syn['neuroTX'],\
                name, ['AMPA', 'GABA', 'NDMA', 'AMPA+NDMA']))
    #* Check no duplicate connections
    if any_duplicates([syn['pre'] + '-' + syn['post'] for syn in synapse_config.values()]):
        raise ConfigException('There are duplicate connections.')
    # if synapse_params is not None:

def encoding_checker(topology):
    """ Checks that the configuration structure and parameters of the encoding ANN 
    configuration are correct.
    """
    for key, encoder in topology['encoding'].items():
        # Check that encoding input exists
        if key not in topology['stimuli']:
            raise ConfigException('Input "{}" of encoder does not exist.'.format(key))
        # Check that encoder exists
        if 'scheme' in encoder and encoder['scheme'] not in reg.encoders:
            raise ConfigException('Encoder "{}" does not exist. '\
                'Available encoders are {}'.format(encoder['scheme'], tuple(reg.encoders)))
        
        #TODO check that sheme available for neuron model.

        # Check receptive field
        if 'receptive_field' not in encoder:
            continue
        rf = encoder['receptive_field']
        if 'name' in rf:
            if rf['name'] not in reg.receptive_fields:
                raise ConfigException('Receptive field "{}" of encoder "{}" is not implemented. '\
                        'Available receptive fields are {}.'\
                        .format(rf['name'], encoder['scheme'], tuple(reg.receptive_fields)))
            rf_base_inspection = inspect.getfullargspec(ReceptiveField)
            rf_child_inspection = inspect.getfullargspec(reg.receptive_fields[rf['name']])
            rf_child_args = {}
            if rf_child_inspection[3] is not None:
                rf_child_args = {key : val for key, val in zip(reversed(rf_child_inspection[0]),\
                                    reversed(rf_child_inspection[3]))}
            if rf_child_inspection[5] is not None:
                 rf_child_args.update(rf_child_inspection[5])
            rf_args = {key : val for key, val in zip(reversed(rf_base_inspection[0]),\
                                    reversed(rf_base_inspection[3]))}
            rf_args = merge_dicts([rf_args, rf_child_args])
            if 'params' in rf:
                for p_name, param in rf['params'].items():
                    if p_name not in rf_args:
                        raise ConfigException('The parameter "{}" does not exist in receptive field "{}" '\
                            'of encoder {}. Available receptive field arguments are {}.'\
                            .format(p_name, rf['name'], key, tuple(rf_args.keys())))
                    valid_types = [type(rf_args[p_name])] if type(rf_args[p_name]) is not float else [float, int]
                    if type(param) not in valid_types:
                        raise ConfigException('Wrong type of parameter "{}" in encoder "{}". Type {} '\
                            'should have been received.'.format(p_name, key, type(rf_args[p_name]).__name__))


def decoding_checker(topology):
    """ Checks that the configuration structure and parameters of the decoding ANN 
    configuration are correct.
    """
    for key, decoder in topology['decoding'].items():
        if key not in topology['outputs']:
            raise ConfigException('Output "{}" of decoder does not exist.'.format(key))
        if 'scheme' in decoder and decoder['scheme'] not in reg.decoders:
            raise ConfigException('Decoder "{}" does not exist. '\
                'Available decoders are {}'.format(decoder['scheme'], tuple(reg.decoders)))
        if 'scheme' not in decoder:
            continue
        dec_base_inspection = inspect.getfullargspec(Decoder)
        dec_child_args = inspect.getfullargspec(reg.decoders[decoder['scheme']]).kwonlydefaults
        dec_args = {key : val for key, val in zip(reversed(dec_base_inspection[0]),\
                        reversed(dec_base_inspection[3]))}
        if dec_child_args is not None:
            dec_args = merge_dicts([dec_child_args, dec_args])
        if 'params' in decoder:
            for p_name, param in decoder['params'].items():
                if p_name not in dec_args:
                    raise ConfigException('The parameter "{}" does not exist in decoder "{}". '\
                            'Available decoder arguments are {}.'\
                            .format(p_name, key, tuple(dec_args.keys())))
                valid_types = [type(dec_args[p_name])] if type(dec_args[p_name]) is not float else [float, int]
                if type(param) not in valid_types:
                    raise ConfigException('Wrong type of parameter "{}" in decoder "{}". Type {} '\
                        'should have been received.'.format(p_name, key, type(dec_args[p_name]).__name__))

def neural_net_checker(topology):
    """ Checks that the provided configuration of the neural network is correct 
    and supported. If error is critical it raises a ConfigException and terminates 
    the execution.
    """
    # Check compulsory variables.
    for var in ['stimuli', 'ensembles', 'synapses', 'outputs']:
        if var not in topology or len(topology[var]) == 0:
            raise ConfigException('Required field {} not specified in config. file.'.format(var))
    if 'dt' in topology and topology['dt'] <= 0.:
        raise ConfigException('ANN Euler step must be greater than 0.')
    if 'time_scale' in topology and topology['time_scale'] <= 0.:
        raise ConfigException('ANN time_scale must be greater than 0.')
    if 'neuron_model' in topology and topology['neuron_model'] not in reg.neuron_models:
        raise ConfigException('Neuron model {} not implemented. '\
            'Available neuron models are {}.'.format(topology['neuron_model'], tuple(reg.neuron_models)))
    if 'synapse_model' in topology and topology['synapse_model'] not in reg.synapse_models:
        raise ConfigException('Synapse model {} not implemented. '\
            'Available synapse models are {}.'.format(topology['synapse_model'], tuple(reg.synapse_models)))

    #* --- Check input stimuli config. ---
    for name, stim in topology['stimuli'].items():
        if 'n' in stim and stim['n'] <= 0.:
            raise ConfigException('Input neurons must be greater than 0. '\
                'Error in input {}.'.format(name))
        if 'sensor' not in stim:
            raise ConfigException('No associated sensors specified in input {}.'\
                .format(name))
        #TODO Check that sensor exists in robot. It requires association between ANN and robot.
        # if stim['sensor'] not in topology
    #* --- Check output config. ---
    for name, out in topology['outputs'].items():
        if 'ensemble' not in out:
            raise ConfigException('No associated ensemble specified in output {}.'\
                .format(name))
        if out['ensemble'] not in topology['ensembles']:
            raise ConfigException('Specified ensemble {} of output {} does not exist.'\
                .format(out['ensemble'], name))
        #TODO Check that actuator exists in robot. It requires association between ANN and robot.
    ensembles_checker(topology)
    synapses_checker(topology)
    if 'encoding' in topology and len(topology['encoding']) and topology['encoding'] is not None:
        encoding_checker(topology)
    if 'decoding' in topology and len(topology['decoding']) and topology['decoding'] is not None:
        decoding_checker(topology)


#* ---- CONFIGURATION AUTOCOMPLETION FUNCTIONS ---- #
def autocomplete_ensembles(topology):
    """ Autocompletion of neural network ensembles and neurons. """
    neuron_model = topology['neuron_model']
    for name, ens in topology['ensembles'].items():
        if 'n' not in ens:
            topology['ensembles'][name]['n'] = 10
            logging.warning('The number of neurons of ensemble {} '\
                'was not settled. Fixing 10 neurons.'.format(name))
        if 'params' not in ens:
            topology['ensembles'][name]['params'] = {}
        neurons_inspect = inspect.getfullargspec(reg.neuron_models[neuron_model].add)
        neuron_args = {key : val for key, val in zip(neurons_inspect.args[1:], neurons_inspect.defaults)}
        for p_name, default_val in neuron_args.items():
            if p_name not in ens['params']:
                topology['ensembles'][name]['params'][p_name] = default_val
    return topology

def autocomplete_synapses(topology):
    """ Autocompletion of neural network synapses and connections. """
    syn_model = topology['synapse_model']
    for name, syn in topology['synapses'].items():
        if 'p' not in syn:
            topology['synapses'][name]['p'] = 1.
            logging.warning('The connection prob. of synapse "{}" '\
                'was not settled. Fixing prob=1.'.format(name))
        for var in ['trainable', 'ntx_fixed']:
            if var not in syn:
                topology['synapses'][name][var] = True
        #TODO: Ampliar a futuros modelos
        if syn_model == 'dynamic_synapse' and 'neuroTX' not in syn:
            topology['synapses'][name]['neuroTX'] = 'AMPA+NDMA'
            logging.warning('No neurotransmitter (neuroTX) of synapse "{}" '\
                'was settled. Fixing neuroTX=AMPA+NDMA.'.format(name))
    #* Autocomplete synapse params.
    default_params = {
        'static_synapse' : {},
        'dynamic_synapse' : {
            "AMPA" : {"gain" : 6., "tau" : 6., "E": None},
            "GABA" : {"gain" : 5., "tau": 10., "E": None},
            "NDMA" : {"gain" : 1., "tau_rise" : 10.0, "tau_decay": 50, "Mg2" : 0.5, "E" : None}
    }}
    if 'synapse_params' not in topology:
        topology['synapse_params'] = default_params[syn_model]
    if syn_model == 'dynamic_synapse':
        for ntx in ['AMPA', 'GABA', 'NDMA']:
            if ntx not in topology['synapse_params']:
                topology['synapse_params'][ntx] = default_params[syn_model][ntx]
            for varname, ntx_var in default_params[syn_model][ntx].items():
                if varname not in topology['synapse_params'][ntx]:
                    topology['synapse_params'][ntx][varname] = ntx_var
    return topology


def autocomplete_encoding(topology):
    """ Autocompletes missing neural configuration related to encoding. 
    """
    is_spiking = issubclass(reg.neuron_models[topology['neuron_model']], SpikingNeuronModel)
    if 'encoding' not in topology:
        topology['encoding'] = {}
    encoding_config = topology['encoding']
    for inpt in topology['stimuli'].keys():
        if inpt not in encoding_config or encoding_config[inpt] is None:
            topology['encoding'][inpt] = {}
            logging.warning('No encoder configured for input "{}". Using default encoder.'\
                .format(inpt))
        if 'scheme' not in encoding_config[inpt]:
            topology['encoding'][inpt]['scheme'] = 'LIF_Encoding' if is_spiking else 'IdentityEncoding'
            logging.warning('No encoding scheme established in "{}". Using encoder "{}".'\
                .format(inpt, topology['encoding'][inpt]['scheme']))
        if 'receptive_field' not in encoding_config[inpt]:
            topology['encoding'][inpt]['receptive_field'] = {}
        if 'name' not in encoding_config[inpt]['receptive_field']:
            topology['encoding'][inpt]['receptive_field']['name'] = 'gaussian_receptive_field'\
                if is_spiking else 'identity_receptive_field'
        if 'params' not in encoding_config[inpt]['receptive_field']:
            # Uses RF class default kwargs.
            topology['encoding'][inpt]['receptive_field']['params'] = {}
    return topology

def autocomplete_decoding(topology):
    """ Autocompletes missing neural configuration related to decoding. 
    """
    is_spiking = issubclass(reg.neuron_models[topology['neuron_model']], SpikingNeuronModel)
    if 'decoding' not in topology:
        topology['decoding'] = {}
    decoding_config = topology['decoding']
    for out in topology['outputs'].keys():
        if out not in decoding_config or decoding_config[out] is None:
            topology['decoding'][out] = {}
            logging.warning('No decoder configured for output "{}". Using default decoder.'.format(out))
        if 'scheme' not in topology['decoding'][out]:
            topology['decoding'][out]['scheme'] = 'LinearPopulationDecoding' if is_spiking else 'IdentityDecoding'
            logging.warning('No decoding scheme established in "{}". Using decoder "{}".'\
                .format(out, topology['decoding'][out]['scheme']))
        if 'params' not in topology['decoding'][out]:
            topology['decoding'][out]['params'] = {}
    return topology

def autocomplete_neural_net(topology):
    """ Autocompletion of neural network configuration. 
    """
    if 'dt' not in topology:
        topology['dt'] = 1.
        logging.warning('No ANN Euler step dt specified. Using dt=1.0')
    if 'time_scale' not in topology:
        topology['time_scale'] = 1
        logging.warning('No ANN time scale specified. Using time_scale=1')
    #TODO autocomplete input dim based on sensor.
    #TODO autocomplete out ensemble dim based on actuator.
    topology = autocomplete_ensembles(topology)
    topology = autocomplete_synapses(topology)
    topology = autocomplete_encoding(topology)
    topology = autocomplete_decoding(topology)
    return topology