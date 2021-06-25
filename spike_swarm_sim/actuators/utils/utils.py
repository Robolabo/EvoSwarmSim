import logging

def list_actuators(robot):
    actuator_list = []
    for name, act in robot.actuators.items():
        if name == 'wireless_transmitter':
            comm_info = ['msg_' + str(i) for i in range(act.msg_length)] + ['state']
            actuator_list.extend(['wireless_transmitter:' + str(key) for key in comm_info])
        elif name == 'wheel_actuator':
            actuator_list.extend(['wheel_actuator_0', 'wheel_actuator_1'])
        elif name in ['joint_position_actuator', 'joint_velocity_actuator']:
            actuator_list.extend([name + '_' + str(i) for i in range(len(robot.actuators[name].joint_ids))])
        else:
            actuator_list.append(name)
    return actuator_list

def check_actuator_cfg(actuator_name, actuator_params):
    if 'min_thresh' in actuator_params and actuator_params['min_thresh'] <= 0:
        raise Exception(logging.error('The min. threshold of the wheels actuator must be greater than 0'))
    if 'dt' in actuator_params and actuator_params['dt'] <= 0:
        raise Exception(logging.error('The Euler step of the wheels actuator must be greater than 0.'))
    if 'range' in actuator_name and actuator_name['range'] <= 0:
        raise Exception(logging.error('The range of actuator {} '\
            'must be greater than 0.'.format(actuator_name)))
    if 'msg_length' in actuator_name and actuator_params['msg_length'] <= 0:
        raise Exception(logging.error('The length of the message of {} '\
            'must be greater than 0.'.format(actuator_name)))

def autocomplete_actuator_cfg(actuator_name, actuator_params):
    if actuator_name == 'wireless_transmitter':
        for var, default in zip(['msg_length', 'range', 'quantize'], [2, 100, True]):
            if var not in actuator_params.keys():
                actuator_params[var] = default
                logging.warning('Parameter {} of Communication Transmitter was not specified. '\
                    'Using default value {}.'.format(var, str(default)))
    elif actuator_name == 'wheel_actuator':
        for var, default in zip(['min_thresh', 'dt'], [0.0, 1.65]):
            if var not in actuator_params.keys():
                actuator_params[var] = default
                logging.warning('Parameter {} of {} was not specified. '\
                    'Using default value {}.'.format(var, actuator_name, str(default)))
    return actuator_params