import logging 
from spike_swarm_sim.register import sensors
from spike_swarm_sim.sensors import DirectionalSensor

def list_sensors(robot):
    sensor_list = []
    for name, sens in robot.sensors.items():
        if name == 'IR_receiver':
            comm_info = ['msg_' + str(i) for i in range(sens.msg_length)]
                # + ['signal', 'sending_direction_x', 'sending_direction_y',\
                # 'receiving_direction_x', 'receiving_direction_y']
            sensor_list.extend(['IR_receiver:' + str(key) for key in comm_info])
        else:
            if hasattr(sens, 'n_sectors'):
                sensor_list.extend([name + '_' + str(i) for i in range(sens.n_sectors)])
            else:
                sensor_list.append(name)
    return sensor_list


def check_sensor_cfg(sensor_name, sensor_params):
    if 'n_sectors' in sensor_params and sensor_params['n_sectors'] <= 0:
        raise Exception(logging.error('The number of sectors of sensor '\
            '{} must be greater than 0.'.format(sensor_name)))
    if 'range' in sensor_params and sensor_params['range'] <= 0:
        raise Exception(logging.error('The range of sensor {} '\
            'must be greater than 0.'.format(sensor_name)))
    if 'msg_length' in sensor_params and sensor_params['msg_length'] <= 0:
        raise Exception(logging.error('The length of the message of {} '\
            'must be greater than 0.'.format(sensor_name)))


def autocomplete_sensor_cfg(sensor_name, sensor_params):
    if sensor_name == 'IR_receiver':
        for var, default in zip(['msg_length', 'range'], [2, 100]):
            if var not in sensor_params.keys():
                sensor_params[var] = default
                logging.warning('Parameter {} of Communication Receiver was not specified. '\
                    'Using default value {}.'.format(var, str(default)), Warning)
    if issubclass(sensors[sensor_name], DirectionalSensor):
        for var, default in zip(['n_sectors', 'range'], [2, 100]):
            if var not in sensor_params.keys():
                sensor_params[var] = default
                logging.warning('Parameter {} of sensor {} was not specified. '\
                    'Using default value {}.'.format(var, sensor_name, str(default)))
    return sensor_params