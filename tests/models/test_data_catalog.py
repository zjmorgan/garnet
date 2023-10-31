from garnet.models import data_catalog
  
import numpy as np
  
login = data_catalog.oncat_login()

#projection = ['facility', 'instrument']

ipts_number = 31189
instrument = 'TOPAZ'
facility = 'SNS', 
tags = ['type/raw']
exts = ['.nxs.h5']

title_entry = 'metadata.entry.title'
run_number_entry = 'metadata.entry.run_number'

ipts_number = 31189
instrument = 'HB3A'
facility = 'HFIR', 
tags = ['type/raw']
exts = ['.dat']

title_entry = 'metadata.scan_title'
run_number_entry = 'metadata.scan'

projection = [title_entry, run_number_entry]

data_files = data_catalog.retrieve_data_files(login,
                                              facility,  
                                              instrument, 
                                              ipts_number,
                                              projection, 
                                              exts, 
                                              tags)

run_title_dict = data_catalog.run_title_dictionary(data_files, 
                                                   title_entry,
                                                   run_number_entry)

print(run_title_dict)

# for key in run_title_dict.keys():
#     print(data_catalog.run_numbers_list(run_title_dict[key]))