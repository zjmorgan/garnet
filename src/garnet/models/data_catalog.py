import pyoncat
import getpass

import numpy as np

def oncat_login():

    ONCAT_URL = 'https://oncat.ornl.gov'
    CLIENT_ID = '99025bb3-ce06-4f4b-bcf2-36ebf925cd1d'

    oncat = pyoncat.ONCat(ONCAT_URL, 
                          client_id=CLIENT_ID,
                          flow=pyoncat.RESOURCE_OWNER_CREDENTIALS_FLOW)

    username = getpass.getuser()

    oncat.login(username,
                getpass.getpass('Enter password for "'+username+'":'))

    return oncat

def goniometer_entries(inst_params):

    goniometer = inst_params['Goniometer']
    goniometer_entry = inst_params['GoniometerEntry']

    projection = []
    for name in goniometer.keys():
        entry = '.'.join([goniometer_entry, name.lower(), 'average_value'])
        projection.append(entry)    
    
    return projection

def retrieve_data_files(login,
                        inst_params, 
                        ipts_number):

    facility = inst_params['Facility']
    instrument = inst_params['Name']

    projection = [inst_params['RunNumber'],
                  inst_params['Title'],
                  inst_params['Scale']]

    projection += goniometer_entries(inst_params)

    exts = [inst_params['Extension']]

    data_files = login.Datafile.list(facility=facility,
                                     instrument=instrument,
                                     experiment='IPTS-{}'.format(ipts_number),
                                     projection=projection,
                                     exts=exts,
                                     tags=['type/raw'])

    return data_files

def run_title_dictionary(data_files, inst_params):

    title_entry = inst_params['Title']
    run_number_entry = inst_params['RunNumber']

    titles = np.array([df[title_entry] for df in data_files])
    run_numbers = np.array([int(df[run_number_entry]) for df in data_files])

    unique_titles = np.unique(titles)

    run_title_dict = {}
    for unique_title in unique_titles:
        runs = run_numbers[titles == unique_title]
        run_seq = np.split(runs.astype(str), np.where(np.diff(runs) > 1)[0]+1)
        rs = ','.join([s[0]+':'+s[-1] if len(s)-1 else s[0] for s in run_seq])
        run_title_dict[unique_title] = rs

    return run_title_dict

def run_numbers_list(rs):

    run_seq = [np.array(s.split(':')).astype(int) for s in rs.split(',')]
    run_list = [np.arange(r[0],r[-1]+1) if len(r)-1 else r for r in run_seq]

    return np.array([r for sub_list in run_list for r in sub_list])