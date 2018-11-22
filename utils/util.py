# Created by Xingyu Lin, 2018/11/19
import numpy as np
import xml.etree.ElementTree as ET
import re
def get_name_arr_and_len(field_indexer, dim_idx):
    """Returns a string array of element names and the max name length."""
    axis = field_indexer._axes[dim_idx]
    size = field_indexer._field.shape[dim_idx]
    try:
        name_len = max(len(name) for name in axis.names)
        name_arr = np.zeros(size, dtype=object)
        for name in axis.names:
            if name:
                # Use the `Axis` object to convert the name into a numpy index, then
                # use this index to write into name_arr.
                name_arr[axis.convert_key_item(name)] = name
    except AttributeError:
        name_arr = np.zeros(size, dtype=object)  # An array of zero-length strings
        name_len = 0
    return name_arr, name_len

def add_rope_actuators(xml_path):
    """Writes XML file with actuators added for rope joints"""
    tree = ET.parse(xml_path)
    doc = tree.getroot()
    ele = doc.find('.//composite[@type="rope"]')
    count_str = ele.attrib['count']
    count = int(re.search(r'\d+', count_str).group())
    actuator_ele = doc.find('actuator')

    for i in range(count):
        if i == int(count/2):
            continue

        motor_name = 'tr' + str(2*i)
        joint_name = 'J0_' + str(i)
        joint1 = ET.Element("motor", ctrllimited="false", name=motor_name, joint=joint_name)

        motor_name = 'tr' + str(2*i + 1)
        joint_name = 'J1_' + str(i)
        joint2 = ET.Element("motor", ctrllimited="false", name=motor_name, joint=joint_name )

        actuator_ele.append(joint1)
        actuator_ele.append(joint2)

    tree.write(xml_path.split('.')[0] + '_temp.xml',encoding='utf8')
    return


